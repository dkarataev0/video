"""Mobile-friendly Gradio UI. Run: python -m studio.app [--share] [--auth user:password]

Long jobs run in a background thread; the page polls the status every 2 s, so the work
continues even if the iPhone locks the screen or Safari reloads the page.
"""
from __future__ import annotations

import argparse
import os
import secrets

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gradio as gr  # noqa: E402

from .config import HOME, MAX_CHARACTERS, MAX_SPEAKERS, SWAPPER_CHOICES  # noqa: E402
from .models import using_gpu  # noqa: E402
from .pipeline import Busy, Project  # noqa: E402

P = Project()

CSS = """
html, body {overflow-x: hidden; max-width: 100vw;}
.gradio-container {max-width: min(760px, 100vw) !important; min-width: 0 !important; width: 100% !important;
                   margin: auto; padding: 0 12px !important; box-sizing: border-box;}
.gradio-container * {min-width: 0;}
.tab-wrapper, .tab-container {flex-wrap: wrap !important; height: auto !important;}
.tab-container button {flex: 0 0 auto;}
#progress {height: 14px; background: #e5e7eb; border-radius: 7px; overflow: hidden;}
#progress > div {height: 100%; background: linear-gradient(90deg,#6366f1,#22c55e); transition: width .5s;}
.slot {border: 1px solid #e5e7eb; border-radius: 12px; padding: 8px;}
button {min-height: 46px;}
"""

CONSENT_TEXT = ("Я подтверждаю, что все люди, чьи лица и голоса используются (и люди в исходном видео), "
                "дали согласие, и что итоговое видео будет распространяться только с пометкой "
                "«Создано с помощью ИИ».")


def _status():
    st = P.state
    pct = int(st.get("progress", 0) * 100)
    icon = {"running": "⏳", "done": "✅", "error": "❌"}.get(st["status"], "ℹ️")
    gpu = "" if using_gpu() else " · ⚠️ GPU не найден — обработка будет очень медленной"
    msg = f"{icon} **{st.get('message', '')}**{gpu}"
    if st["status"] == "error" and st.get("error"):
        msg += f"\n\n<details><summary>Подробности</summary>\n\n```\n{st['error'][:2000]}\n```\n</details>"
    bar = f'<div id="progress"><div style="width:{pct}%"></div></div><div style="text-align:right">{pct}%</div>'
    return msg, bar


def _rev():
    st = P.state
    return str((st["status"], st.get("job"), len(st["characters"]), len(st["speakers"]), st.get("result"),
                st.get("test"), st.get("cluster_threshold"), len(st["segments"]),
                sum(1 for _ in st["face_photos"]), sum(1 for _ in st["voice_files"])))


def _panels():
    st = P.state
    ups = []
    for i in range(MAX_CHARACTERS):
        c = st["characters"][i] if i < len(st["characters"]) else None
        if c is None:
            ups += [gr.update(visible=False), gr.update(value=None), gr.update(value="")]
        else:
            n = len(st["face_photos"].get(str(i), []))
            label = f"**Персонаж {i + 1}** — в кадре ≈ {c['seconds']} с. " + (
                f"Назначено фото: {n} ✅" if n else "Не заменяется (загрузите фото, чтобы заменить)")
            ups += [gr.update(visible=True), gr.update(value=c.get("thumb")), gr.update(value=label)]
    for i in range(MAX_SPEAKERS):
        s = st["speakers"][i] if i < len(st["speakers"]) else None
        if s is None:
            ups += [gr.update(visible=False), gr.update(value=None), gr.update(value="")]
        else:
            n = len(st["voice_files"].get(str(i), []))
            label = f"**Голос {i + 1}** — говорит ≈ {s['seconds']} с. " + (
                f"Назначен образец ✅ ({n} файл.)" if n else "Не заменяется (загрузите образец голоса)")
            ups += [gr.update(visible=True), gr.update(value=s.get("preview")), gr.update(value=label)]
    seg_rows = [[s["start"], s["end"], s["speaker"] + 1] for s in st["segments"]]
    return ups + [gr.update(value=seg_rows), gr.update(value=st.get("test")),
                  gr.update(value=st.get("result")), gr.update(value=st.get("result"))]


def refresh(seen):
    msg, bar = _status()
    rev = _rev()
    if rev == seen:
        return [msg, bar, seen] + [gr.skip()] * N_PANEL_OUT
    return [msg, bar, rev] + _panels()


def _guard(consent):
    if not consent:
        raise gr.Error("Подтвердите согласие участников (галочка в начале страницы).")


def on_video(file, consent):
    _guard(consent)
    if file is None:
        return
    try:
        P.set_video(file)
        P.state["consent"] = True
        P.start("analyze", P.analyze_job)
    except Busy as e:
        raise gr.Error(str(e))
    gr.Info("Видео загружено, начат анализ. Можно заблокировать телефон — работа продолжится.")


def on_face(i, clear=False):
    def f(files=None):
        if P.running:
            raise gr.Error("Дождитесь окончания текущей задачи, затем меняйте фото.")
        P.set_face_photos(i, None if clear else files)
        if not clear:
            gr.Info(f"Фото для персонажа {i + 1} сохранены.")
    return f


def on_voice(i, clear=False):
    def f(files=None):
        if P.running:
            raise gr.Error("Дождитесь окончания текущей задачи, затем меняйте образцы.")
        P.set_voice_files(i, None if clear else files)
        if not clear:
            gr.Info(f"Образец для голоса {i + 1} сохранён.")
    return f


def on_recluster(th):
    if not P.meta:
        raise gr.Error("Сначала загрузите видео.")
    if P.running:
        raise gr.Error("Дождитесь окончания задачи.")
    P.recluster(float(th))
    gr.Info("Персонажи пересчитаны. Назначьте фото заново.")


def on_rediarize(n):
    if not P.state.get("vocals"):
        raise gr.Error("В видео нет звука или анализ ещё не завершён.")
    try:
        P.start("diarize", P.rediarize, int(n) or None, 0.6)
    except Busy as e:
        raise gr.Error(str(e))


def on_segments(rows):
    segs = []
    try:
        values = rows.values.tolist() if hasattr(rows, "values") else rows
        for r in values:
            if r is None or len(r) < 3 or r[0] in ("", None):
                continue
            segs.append({"start": float(r[0]), "end": float(r[1]), "speaker": int(float(r[2])) - 1})
    except Exception:
        raise gr.Error("Ошибка в таблице: начало/конец — секунды, говорящий — номер (1, 2, …).")
    if any(s["speaker"] < 0 or s["speaker"] >= MAX_SPEAKERS for s in segs):
        raise gr.Error(f"Номер говорящего должен быть от 1 до {MAX_SPEAKERS}.")
    P.set_segments(segs)
    gr.Info("Разметка сохранена.")


def _apply_settings(*vals):
    keys = ["swapper", "pixel_boost", "face_weight", "enhancer_blend", "occlusion_mask", "deflicker",
            "vc_steps", "vc_keep_intonation", "vc_semitones", "clean_reference", "label_position"]
    kw = dict(zip(keys, vals))
    kw["pixel_boost"] = int(kw["pixel_boost"])
    kw["vc_steps"] = int(kw["vc_steps"])
    kw["vc_semitones"] = int(kw["vc_semitones"])
    P.set_settings(**kw)


def _ready(consent):
    _guard(consent)
    if not P.meta or not P.state["characters"] and not P.state["speakers"]:
        raise gr.Error("Сначала загрузите видео и дождитесь окончания анализа.")
    if not P.state["face_photos"] and not P.state["voice_files"]:
        raise gr.Error("Назначьте хотя бы одно лицо или один голос.")


def on_test(consent, start, *settings):
    _ready(consent)
    _apply_settings(*settings)
    try:
        P.start("test", P.test_job, float(start or 0))
    except Busy as e:
        raise gr.Error(str(e))


def on_full(consent, *settings):
    _ready(consent)
    _apply_settings(*settings)
    try:
        P.start("full", P.full_job)
    except Busy as e:
        raise gr.Error(str(e))


def on_regen(consent, t0, t1, what, *settings):
    _ready(consent)
    if not P.state.get("result"):
        raise gr.Error("Сначала выполните полную обработку.")
    t0, t1 = float(t0 or 0), float(t1 or 0)
    if t1 <= t0:
        raise gr.Error("Конец фрагмента должен быть больше начала.")
    _apply_settings(*settings)
    mode = {"Лица и голоса": "all", "Только лица": "faces", "Только голоса": "voices"}[what]
    try:
        P.start("regen", P.full_job, (t0, t1), mode)
    except Busy as e:
        raise gr.Error(str(e))


def on_wipe(confirm):
    if not confirm:
        raise gr.Error("Поставьте галочку подтверждения удаления.")
    try:
        P.wipe()
    except Busy as e:
        raise gr.Error(str(e))
    gr.Info("Все загруженные фото, голоса, видео и результаты удалены с сервера.")


def build():
    global N_PANEL_OUT
    s = P.state["settings"]
    with gr.Blocks(title="AI Video Studio", css=CSS, theme=gr.themes.Soft(font=["-apple-system", "system-ui", "Segoe UI", "Roboto", "sans-serif"],
                                                                    font_mono=["ui-monospace", "Menlo", "monospace"])) as demo:
        gr.Markdown("## 🎬 AI Video Studio\nЗамена лиц и голосов с согласия участников. "
                    "Итог всегда помечается как созданный ИИ.")
        consent = gr.Checkbox(label=CONSENT_TEXT, value=bool(P.state.get("consent")))
        status = gr.Markdown()
        bar = gr.HTML()
        seen = gr.State("")

        with gr.Tab("1. Видео"):
            video_in = gr.File(label="Исходное видео (MP4/MOV, из «Фото» или «Файлов»)", file_types=["video"])
            gr.Markdown("После загрузки автоматически запустится анализ: поиск персонажей, "
                        "отделение голосов от музыки и определение говорящих.")

        char_groups, char_imgs, char_lbls = [], [], []
        with gr.Tab("2. Лица"):
            gr.Markdown("Для каждого персонажа загрузите 1–10 фото человека, на которого его заменить "
                        "(лицо крупно, хороший свет, разные ракурсы — лучше результат). "
                        "Если персонажа не нужно менять — ничего не загружайте.")
            for i in range(MAX_CHARACTERS):
                with gr.Group(visible=False, elem_classes="slot") as g:
                    with gr.Row():
                        img = gr.Image(label=f"Персонаж {i + 1}", interactive=False, height=150,
                                       show_download_button=False, min_width=120)
                        with gr.Column(min_width=180):
                            lbl = gr.Markdown()
                            up = gr.File(label="Фото нового лица", file_count="multiple", file_types=["image"])
                    up.upload(on_face(i), up, None)
                    up.clear(on_face(i, clear=True), None, None)
                char_groups.append(g)
                char_imgs.append(img)
                char_lbls.append(lbl)
            with gr.Accordion("Один человек разбит на несколько персонажей / разные люди слились?", open=False):
                th = gr.Slider(0.3, 0.8, value=0.55, step=0.05,
                               label="Порог объединения (меньше — больше персонажей, больше — меньше)")
                gr.Button("Пересчитать персонажей").click(on_recluster, th, None)
                gr.Markdown("Подсказка: если один человек оказался в двух карточках — просто загрузите "
                            "одни и те же фото в обе.")

        spk_groups, spk_audio, spk_lbls = [], [], []
        with gr.Tab("3. Голоса"):
            gr.Markdown("Прослушайте найденные голоса и загрузите для нужных образец нового голоса "
                        "(10–30 с чистой речи; подойдёт диктофонная запись .m4a или видео).")
            for i in range(MAX_SPEAKERS):
                with gr.Group(visible=False, elem_classes="slot") as g:
                    lbl = gr.Markdown()
                    au = gr.Audio(label=f"Как звучит голос {i + 1} в видео", interactive=False, type="filepath")
                    up = gr.File(label="Образец нового голоса", file_count="multiple", file_types=["audio", "video"])
                    up.upload(on_voice(i), up, None)
                    up.clear(on_voice(i, clear=True), None, None)
                spk_groups.append(g)
                spk_audio.append(au)
                spk_lbls.append(lbl)
            with gr.Accordion("Исправить, кто когда говорит", open=False):
                nspk = gr.Number(label="Сколько разных говорящих (0 = определить автоматически)", value=0,
                                 precision=0)
                gr.Button("Пересчитать говорящих").click(on_rediarize, nspk, None)
                segtab = gr.Dataframe(headers=["Начало, с", "Конец, с", "Говорящий №"],
                                      datatype=["number", "number", "number"], interactive=True,
                                      col_count=(3, "fixed"), label="Разметка речи (можно править)")
                gr.Button("Сохранить разметку").click(on_segments, segtab, None)

        with gr.Tab("4. Настройки"):
            swapper = gr.Dropdown(SWAPPER_CHOICES, value=s["swapper"], label="Модель замены лица")
            boost = gr.Radio([256, 512, 768], value=s["pixel_boost"], label="Детализация лица (512 — оптимум)")
            weight = gr.Slider(0, 1, value=s["face_weight"], step=0.05,
                               label="Сила сходства с новым лицом (0.5 — по умолчанию)")
            enh = gr.Slider(0, 100, value=s["enhancer_blend"], step=5, label="Улучшение лица GFPGAN, % (0 — выкл.)")
            occl = gr.Checkbox(value=s["occlusion_mask"], label="Учитывать руки/волосы/очки перед лицом (XSeg)")
            defl = gr.Slider(0, 1, value=s["deflicker"], step=0.1, label="Подавление мерцания")
            steps = gr.Slider(10, 50, value=s["vc_steps"], step=5, label="Качество голоса (шаги диффузии)")
            inton = gr.Checkbox(value=s["vc_keep_intonation"],
                                label="Точно сохранять интонацию исходной речи (модель с F0, 44 кГц)")
            semi = gr.Slider(-12, 12, value=s["vc_semitones"], step=1, label="Сдвиг высоты голоса, полутоны")
            clean = gr.Checkbox(value=s["clean_reference"], label="Очищать образцы голоса от музыки/шума")
            lpos = gr.Dropdown(["bottom-right", "bottom-left", "top-right", "top-left"], value=s["label_position"],
                               label="Положение пометки «Создано с помощью ИИ»")
        settings = [swapper, boost, weight, enh, occl, defl, steps, inton, semi, clean, lpos]

        with gr.Tab("5. Результат"):
            with gr.Row():
                tstart = gr.Number(label="Тест: начало фрагмента, с", value=0)
                btest = gr.Button("▶️ Тест 5 секунд", variant="secondary")
            test_vid = gr.Video(label="Тестовый фрагмент", interactive=False)
            bfull = gr.Button("🚀 Полная обработка", variant="primary")
            res_vid = gr.Video(label="Результат", interactive=False)
            res_file = gr.File(label="Скачать MP4", interactive=False)
            with gr.Accordion("Перегенерировать фрагмент", open=False):
                with gr.Row():
                    r0 = gr.Number(label="С секунды", value=0)
                    r1 = gr.Number(label="По секунду", value=5)
                what = gr.Radio(["Лица и голоса", "Только лица", "Только голоса"], value="Лица и голоса",
                                label="Что пересчитать")
                gr.Markdown("Можно поменять настройки или фото — они применятся только к этому фрагменту.")
                bregen = gr.Button("🔁 Перегенерировать")

        with gr.Tab("🔒 Данные"):
            gr.Markdown("Файлы хранятся только во временной папке вашей сессии Kaggle и удаляются при её "
                        "завершении. Здесь можно удалить их немедленно.")
            conf = gr.Checkbox(label="Да, удалить всё")
            gr.Button("🗑 Удалить все мои данные", variant="stop").click(on_wipe, conf, None)

        panel_out = []
        for g, im, lb in zip(char_groups, char_imgs, char_lbls):
            panel_out += [g, im, lb]
        for g, au, lb in zip(spk_groups, spk_audio, spk_lbls):
            panel_out += [g, au, lb]
        panel_out += [segtab, test_vid, res_vid, res_file]
        N_PANEL_OUT = len(panel_out)

        video_in.upload(on_video, [video_in, consent], None)
        btest.click(on_test, [consent, tstart] + settings, None)
        bfull.click(on_full, [consent] + settings, None)
        bregen.click(on_regen, [consent, r0, r1, what] + settings, None)
        timer = gr.Timer(2.0)
        timer.tick(refresh, seen, [status, bar, seen] + panel_out, show_progress="hidden")
        demo.load(refresh, seen, [status, bar, seen] + panel_out)
    return demo


N_PANEL_OUT = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public *.gradio.live link (for iPhone)")
    ap.add_argument("--auth", default=os.environ.get("STUDIO_AUTH", ""), help="user:password")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    if a.auth:
        user, pwd = a.auth.split(":", 1)
    else:
        user, pwd = "studio", secrets.token_urlsafe(8)
    print("=" * 60)
    print(f"ЛОГИН: {user}\nПАРОЛЬ: {pwd}")
    print("=" * 60, flush=True)
    demo = build()
    demo.queue(default_concurrency_limit=4).launch(
        share=a.share, server_name="0.0.0.0", server_port=a.port, auth=(user, pwd),
        allowed_paths=[str(HOME)], max_file_size="2gb", show_api=False)


if __name__ == "__main__":
    main()
