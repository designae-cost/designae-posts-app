import os
import io
import base64
import random
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas

# ──────────────────────────────────────────────────────────────────────
# 1. STREAMLIT PAGE SETUP
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DESIGNAE Posts Generator", layout="centered")
st.title("DESIGNAE Posts Generator")

TAMANHO_FONTE_PX = 35
ENTRELINHA_PX    = 40

# Logical post dimensions and export scale
POST_W, POST_H = 1080, 1350
EXPORT_SCALE = 2
EXPORT_W = POST_W * EXPORT_SCALE
EXPORT_H = POST_H * EXPORT_SCALE

# Preview canvas dimensions
CANVAS_W = 540
CANVAS_H = 675
ESCALA   = CANVAS_W / POST_W   # 0.5

TEMPLATE_BLEND_OPACITY = 0.75


def carregar_fonte(pasta_assets, tamanho=TAMANHO_FONTE_PX):
    caminhos_fonte = [
        os.path.join(pasta_assets, "fonte_projeto.ttf"),
        os.path.join(pasta_assets, "fonts", "Montserrat-Bold.ttf"),
        os.path.join(pasta_assets, "Montserrat-Bold.ttf"),
        "Arial.ttf"
    ]
    for caminho in caminhos_fonte:
        try:
            if os.path.exists(caminho):
                return ImageFont.truetype(caminho, tamanho)
        except Exception:
            continue
    try:
        return ImageFont.truetype("Arial", tamanho)
    except Exception:
        return ImageFont.load_default()


def quebrar_texto_estrito(texto, max_caracteres=18):
    paragrafos = texto.split("\n")
    linhas_finais = []
    for paragrafo in paragrafos:
        if not paragrafo.strip():
            linhas_finais.append("")
            continue
        palavras = paragrafo.split()
        linha_atual = []
        for palavra in palavras:
            test_linha = " ".join(linha_atual + [palavra])
            if len(test_linha) <= max_caracteres:
                linha_atual.append(palavra)
            else:
                if s_line := " ".join(linha_atual):
                    linhas_finais.append(s_line)
                linha_atual = [palavra]
        if linha_atual:
            linhas_finais.append(" ".join(linha_atual))
    return linhas_finais


def carregar_mini_bars(pasta_assets, render_scale):
    path_minibars = os.path.join(pasta_assets, "mini_bars_anchor.png")
    if not os.path.exists(path_minibars):
        return None

    mini = Image.open(path_minibars).convert("RGBA")
    fator = render_scale / EXPORT_SCALE
    largura = max(1, int(round(mini.width * fator)))
    altura = max(1, int(round(mini.height * fator)))
    if largura == mini.width and altura == mini.height:
        return mini
    return mini.resize((largura, altura), Image.Resampling.LANCZOS)


def suavizar_perfil_vertical(perfil, passes=4):
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel /= kernel.sum()
    resultado = perfil.astype(np.float32)

    for _ in range(passes):
        padded = np.pad(resultado, ((2, 2), (0, 0)), mode="edge")
        canais = [
            np.convolve(padded[:, idx], kernel, mode="valid")
            for idx in range(resultado.shape[1])
        ]
        resultado = np.stack(canais, axis=1)

    return np.clip(resultado, 0.0, 1.0)


def gerar_factor_gradiente(path_gradiente, largura, altura):
    grad = Image.open(path_gradiente).convert("RGBA")
    prop = largura / grad.width
    grad = grad.resize((largura, int(grad.height * prop)), Image.Resampling.LANCZOS)
    grad_arr = np.asarray(grad, dtype=np.float32) / 255.0

    perfil = grad_arr.mean(axis=1)
    perfil = suavizar_perfil_vertical(perfil)

    layer = np.ones((altura, 4), dtype=np.float32)
    layer[:, 3] = 0.0
    start_y = max(0, altura - perfil.shape[0])
    layer[start_y:, :] = perfil[-(altura - start_y):, :]

    grad_rgb = layer[:, :3]
    grad_alpha = layer[:, 3:4]
    factor_1d = (1.0 - grad_alpha) + (grad_alpha * grad_rgb)
    return np.broadcast_to(factor_1d[:, None, :], (altura, largura, 3))


def preparar_overlay_rgba(imagem_rgba, largura, altura):
    src_w, src_h = imagem_rgba.size

    if src_w == largura and src_h == altura:
        return imagem_rgba

    if abs(src_w - largura) <= 2 and abs(src_h - altura) <= 2:
        left = max(0, (src_w - largura) // 2)
        top = max(0, (src_h - altura) // 2)
        right = min(src_w, left + largura)
        bottom = min(src_h, top + altura)
        cropped = imagem_rgba.crop((left, top, right, bottom))

        if cropped.size == (largura, altura):
            return cropped

        canvas = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
        paste_x = max(0, (largura - cropped.width) // 2)
        paste_y = max(0, (altura - cropped.height) // 2)
        canvas.paste(cropped, (paste_x, paste_y), cropped)
        return canvas

    return imagem_rgba.resize((largura, altura), Image.Resampling.LANCZOS)


def aplicar_multiply_rgba(base_rgb, overlay_rgba, opacity=TEMPLATE_BLEND_OPACITY):
    base_arr = np.asarray(base_rgb, dtype=np.float32) / 255.0
    overlay_arr = np.asarray(overlay_rgba, dtype=np.float32) / 255.0

    overlay_rgb = overlay_arr[..., :3]
    overlay_alpha = overlay_arr[..., 3:4] * opacity
    multiplied = base_arr * overlay_rgb
    resultado = (base_arr * (1.0 - overlay_alpha)) + (multiplied * overlay_alpha)

    return Image.fromarray(
        np.clip(resultado * 255.0, 0, 255).astype(np.uint8),
        mode="RGB",
    )


def listar_assets_templates(pasta_assets):
    nomes = [f"Template_{idx}.png" for idx in range(1, 7)]
    return [nome for nome in nomes if os.path.exists(os.path.join(pasta_assets, nome))]


def escolher_template_aleatorio(pasta_assets):
    assets = listar_assets_templates(pasta_assets)
    if not assets:
        return None
    return random.choice(assets)


def aplicar_template_overlay(foto_rgb, largura, altura, pasta_assets, template_nome):
    if not template_nome:
        return foto_rgb

    path_template = os.path.join(pasta_assets, template_nome)
    if not os.path.exists(path_template):
        return foto_rgb

    template = Image.open(path_template).convert("RGBA")
    template = preparar_overlay_rgba(template, largura, altura)
    return aplicar_multiply_rgba(foto_rgb, template)


def gerar_post_designae(imagem_cortada, texto_orador, texto_x, texto_y, template_nome, desenhar_texto=True):
    LARGURA, ALTURA = EXPORT_W, EXPORT_H
    pasta_assets = "assets"

    foto_base = imagem_cortada.convert("RGBA").resize((LARGURA, ALTURA), Image.Resampling.LANCZOS).convert("RGB")
    foto_base = aplicar_template_overlay(foto_base, LARGURA, ALTURA, pasta_assets, template_nome)

    path_gradiente = os.path.join(pasta_assets, "barra_gradiente.png")
    path_logos     = os.path.join(pasta_assets, "barra_logos.png")

    if os.path.exists(path_gradiente):
        foto_arr = np.asarray(foto_base, dtype=np.float32) / 255.0
        factor = gerar_factor_gradiente(path_gradiente, LARGURA, ALTURA)
        foto_base = Image.fromarray(
            np.clip(foto_arr * factor * 255.0, 0, 255).astype(np.uint8),
            mode="RGB",
        )

    camada_fixos = Image.new("RGBA", (LARGURA, ALTURA), (0, 0, 0, 0))

    if os.path.exists(path_logos):
        logos = Image.open(path_logos).convert("RGBA")
        logos = preparar_overlay_rgba(logos, LARGURA, ALTURA)
        camada_fixos.paste(logos, (0, 0), logos)

    if desenhar_texto and texto_orador.strip():
        render_scale = EXPORT_SCALE
        fonte = carregar_fonte(pasta_assets, tamanho=TAMANHO_FONTE_PX * render_scale)
        linhas = quebrar_texto_estrito(texto_orador, max_caracteres=18)

        origem_x = int(round(texto_x * render_scale))
        origem_y = int(round(texto_y * render_scale))
        recuo_texto_x = 0

        mini = carregar_mini_bars(pasta_assets, render_scale)
        if mini is not None:
            largura_m = mini.width
            camada_fixos.paste(mini, (origem_x, origem_y), mini)
            recuo_texto_x = largura_m - (10 * render_scale)

        draw = ImageDraw.Draw(camada_fixos)
        y_linha_atual = origem_y + (40 * render_scale)

        for linha in linhas:
            if linha.strip():
                draw.text(
                    (origem_x + recuo_texto_x, y_linha_atual),
                    linha,
                    fill=(255, 255, 255, 255),
                    font=fonte,
                )
            y_linha_atual += ENTRELINHA_PX * render_scale

    resultado = Image.alpha_composite(foto_base.convert("RGBA"), camada_fixos)
    return resultado.convert("RGB")


def imagem_para_data_url(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def calcular_caixa_texto(texto_orador):
    """Calcula a área arrastável do bloco de texto na escala real do post."""
    pasta_assets = "assets"
    fonte = carregar_fonte(pasta_assets, tamanho=TAMANHO_FONTE_PX)
    linhas = quebrar_texto_estrito(texto_orador, max_caracteres=18)

    largura_texto = 180
    if linhas:
        medidor = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        larguras = []
        for linha in linhas:
            if linha.strip():
                bbox = medidor.textbbox((0, 0), linha, font=fonte)
                larguras.append(bbox[2] - bbox[0])
        if larguras:
            largura_texto = max(larguras)

    largura_mini = 0
    altura_mini = 0
    mini = carregar_mini_bars(pasta_assets, 1)
    if mini is not None:
        largura_mini, altura_mini = mini.size

    recuo_texto_x = largura_mini + -10 if largura_mini else 0
    largura = max(largura_mini, recuo_texto_x + largura_texto) + 30
    altura_texto = 40 + max(len(linhas), 1) * ENTRELINHA_PX
    altura = max(altura_mini, altura_texto) + 20

    return max(largura, 180), max(altura, 90)


def gerar_bloco_texto_preview(texto_orador, largura, altura):
    bloco = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))

    if texto_orador.strip():
        pasta_assets = "assets"
        recuo_texto_x = 0

        mini = carregar_mini_bars(pasta_assets, 1)
        if mini is not None:
            bloco.paste(mini, (0, 0), mini)
            recuo_texto_x = mini.width + -10

        fonte = carregar_fonte(pasta_assets, tamanho=TAMANHO_FONTE_PX)
        linhas = quebrar_texto_estrito(texto_orador, max_caracteres=18)
        draw = ImageDraw.Draw(bloco)
        y_linha_atual = 40

        for linha in linhas:
            if linha.strip():
                draw.text(
                    (recuo_texto_x, y_linha_atual),
                    linha,
                    fill=(255, 255, 255, 255),
                    font=fonte,
                )
            y_linha_atual += ENTRELINHA_PX

    return bloco


def preparar_bloco_arrastavel(texto_orador, largura, altura, escala):
    bloco = gerar_bloco_texto_preview(texto_orador, largura, altura)
    preview_w = max(60, int(largura * escala))
    preview_h = max(45, int(altura * escala))
    return bloco.resize((preview_w, preview_h), Image.Resampling.LANCZOS)


def desenho_bloco_texto(x, y, bloco_preview):
    largura, altura = bloco_preview.size
    return {
        "version": "4.4.0",
        "objects": [
            {
                "type": "image",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": x,
                "top": y,
                "width": largura,
                "height": altura,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "opacity": 1,
                "src": imagem_para_data_url(bloco_preview),
                "selectable": True,
                "evented": True,
                "hasControls": False,
                "hasBorders": True,
                "borderColor": "#FF4B4B",
                "cornerColor": "#FF4B4B",
                "transparentCorners": False,
                "lockScalingX": True,
                "lockScalingY": True,
                "lockRotation": True,
                "hoverCursor": "move",
                "moveCursor": "move",
            }
        ],
    }


def desenho_preview_canvas(imagem_preview, x, y, bloco_preview):
    largura_bg, altura_bg = imagem_preview.size
    largura_bloco, altura_bloco = bloco_preview.size

    return {
        "version": "4.4.0",
        "objects": [
            {
                "type": "image",
                "version": "4.4.0",
                "name": "preview_background",
                "originX": "left",
                "originY": "top",
                "left": 0,
                "top": 0,
                "width": largura_bg,
                "height": altura_bg,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "opacity": 1,
                "src": imagem_para_data_url(imagem_preview),
                "selectable": False,
                "evented": False,
                "hasControls": False,
                "hasBorders": False,
                "lockMovementX": True,
                "lockMovementY": True,
                "hoverCursor": "default",
                "moveCursor": "default",
            },
            {
                "type": "image",
                "version": "4.4.0",
                "name": "text_block",
                "originX": "left",
                "originY": "top",
                "left": x,
                "top": y,
                "width": largura_bloco,
                "height": altura_bloco,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "opacity": 1,
                "src": imagem_para_data_url(bloco_preview),
                "selectable": True,
                "evented": True,
                "hasControls": False,
                "hasBorders": True,
                "borderColor": "#FF4B4B",
                "cornerColor": "#FF4B4B",
                "transparentCorners": False,
                "lockScalingX": True,
                "lockScalingY": True,
                "lockRotation": True,
                "hoverCursor": "move",
                "moveCursor": "move",
            },
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# 2. INITIALIZE SESSION STATE
# ──────────────────────────────────────────────────────────────────────
if "pos_x" not in st.session_state:
    st.session_state.pos_x = 140
if "pos_y" not in st.session_state:
    st.session_state.pos_y = 680
if "template_nome" not in st.session_state:
    st.session_state.template_nome = escolher_template_aleatorio("assets")
if "export_image_bytes" not in st.session_state:
    st.session_state.export_image_bytes = None


# ──────────────────────────────────────────────────────────────────────
# 3. USER INTERFACE
# ──────────────────────────────────────────────────────────────────────
file_carregado = st.file_uploader("Choose background image (.jpg, .png)", type=["jpg", "jpeg", "png"])

if file_carregado is not None:
    imagem_original = Image.open(file_carregado)

    st.markdown("### Crop Image")
    imagem_cortada = st_cropper(
        imagem_original,
        aspect_ratio=(4, 5),
        box_color="#FF4B4B",
        return_type="image"
    )
    st.markdown("### Caption Text")
    texto_orador = st.text_area("", value="", label_visibility="collapsed")
    if st.button("Change Colours"):
        st.session_state.template_nome = escolher_template_aleatorio("assets")

    # Build the preview background without text; the text stays in a draggable layer.
    img_preview_base = gerar_post_designae(
        imagem_cortada,
        texto_orador,
        0,
        0,
        st.session_state.template_nome,
        desenhar_texto=False
    ).resize((CANVAS_W, CANVAS_H), Image.Resampling.LANCZOS)

    # Draggable text block in preview scale.
    caixa_w, caixa_h = calcular_caixa_texto(texto_orador)
    caixa_x = int(st.session_state.pos_x * ESCALA)
    caixa_y = int(st.session_state.pos_y * ESCALA)
    bloco_preview = preparar_bloco_arrastavel(texto_orador, caixa_w, caixa_h, ESCALA)

    st.markdown("### Post Preview")
    st.caption(
        "Drag the text to reposition the text block."
    )

    current_pos_x = st.session_state.pos_x
    current_pos_y = st.session_state.pos_y

    canvas_result = st_canvas(
        fill_color      ="rgba(255, 75, 75, 0.01)",
        stroke_width    =2,
        stroke_color    ="#FF4B4B",
        background_color="#FFFFFF",
        drawing_mode    ="transform",
        initial_drawing =desenho_preview_canvas(
            img_preview_base,
            caixa_x,
            caixa_y,
            bloco_preview,
        ),
        display_toolbar =False,
        update_streamlit=True,
        height=CANVAS_H,
        width =CANVAS_W,
        key   ="canvas_pos",
    )

    # Read the draggable block and update the position from its top-left corner.
    if canvas_result.json_data is not None:
        objectos = canvas_result.json_data.get("objects", [])
        caixa = next(
            (obj for obj in objectos if obj.get("name") == "text_block"),
            objectos[-1] if objectos else None,
        )
        if caixa is not None:
            cx = float(caixa.get("left", caixa_x))
            cy = float(caixa.get("top",  caixa_y))

            novo_x = int(round(cx / ESCALA))
            novo_y = int(round(cy / ESCALA))
            novo_x = min(max(novo_x, 0), POST_W)
            novo_y = min(max(novo_y, 0), POST_H)
            current_pos_x = novo_x
            current_pos_y = novo_y
            if (
                novo_x != st.session_state.pos_x
                or novo_y != st.session_state.pos_y
            ):
                st.session_state.pos_x = novo_x
                st.session_state.pos_y = novo_y
                st.rerun()

    # Final image download button.
    img_final = gerar_post_designae(
        imagem_cortada,
        texto_orador,
        st.session_state.pos_x,
        st.session_state.pos_y,
        st.session_state.template_nome
    )

    buf = io.BytesIO()
    img_final.save(buf, format="PNG")
    st.session_state.export_image_bytes = buf.getvalue()
    st.download_button(
        label="Download Image",
        data=st.session_state.export_image_bytes,
        file_name="post_designae.png",
        mime="image/png",
        on_click="ignore",
    )
else:
    st.info("Please upload a background image to preview the post.")
