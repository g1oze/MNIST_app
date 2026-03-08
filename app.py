import numpy as np
import streamlit as st
from PIL import Image

from model import Predictor
from streamlit_drawable_canvas import st_canvas


def probability_to_color(probability: float) -> str:
    p = max(0.0, min(1.0, float(probability)))
    r = int(239 - 181 * p)
    g = int(68 + 120 * p)
    b = int(68 + 19 * p)
    return f"rgb({r}, {g}, {b})"


def parse_input_data(raw_text: str) -> np.ndarray:
    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("Введите хотя бы одну строку с данными.")

    rows = []
    for idx, line in enumerate(lines, start=1):
        values = [x.strip() for x in line.split(",") if x.strip()]
        if len(values) != 784:
            raise ValueError(
                f"Строка {idx}: ожидается 784 значения через запятую, получено {len(values)}."
            )
        try:
            rows.append([float(v) for v in values])
        except ValueError as exc:
            raise ValueError(f"Строка {idx}: найдены нечисловые значения.") from exc

    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    st.set_page_config(page_title="MNIST Predictor", layout="centered")
    st.title("MNIST Predictor")

    st.subheader("Параметры модели")
    weights_path = st.text_input("Путь к файлу весов", value="model_weights.pth")
    device = st.selectbox("Устройство", options=["cpu", "cuda"], index=0)

    st.subheader("Входные данные")
    input_mode = st.radio("Режим ввода", ["Рисунок", "Ручной ввод"], horizontal=True)

    drawn_input = None
    if input_mode == "Рисунок":
        st.caption("Нарисуйте цифру белым цветом на черном фоне и нажмите 'Рассчитать'.")
        if "canvas_key_suffix" not in st.session_state:
            st.session_state.canvas_key_suffix = 0

        controls_col1, controls_col2 = st.columns([1, 1])
        with controls_col1:
            stroke_width = st.slider("Толщина линии", min_value=6, max_value=30, value=16)
        with controls_col2:
            st.write("")
            if st.button("Очистить холст"):
                st.session_state.canvas_key_suffix += 1
                st.rerun()

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key=f"mnist_canvas_{st.session_state.canvas_key_suffix}",
        )

        if canvas_result.image_data is not None:
            rgba = canvas_result.image_data
            grayscale = rgba[:, :, 0].astype(np.float32) / 255.0
            grayscale_uint8 = np.clip(grayscale * 255.0, 0, 255).astype(np.uint8)
            resized = Image.fromarray(grayscale_uint8).resize((28, 28), Image.Resampling.BILINEAR)
            pooled = np.asarray(resized, dtype=np.float32) / 255.0
            drawn_input = pooled.reshape(1, 784).astype(np.float32)
            st.image(pooled, caption="Предобработанное изображение 28x28", clamp=True, width=180)
    else:
        st.caption(
            "Введите данные: 1 строка = 1 изображение, в каждой строке ровно 784 числа через запятую."
        )
        default_input = ",".join(["0"] * 784)
        data_text = st.text_area("Данные", value=default_input, height=220)

    if st.button("Рассчитать", type="primary"):
        try:
            if input_mode == "Рисунок":
                if drawn_input is None:
                    raise ValueError("Сначала нарисуйте цифру на холсте.")
                data = drawn_input
            else:
                data = parse_input_data(data_text)
            predictor = Predictor(weights_path=weights_path, device=device)
            probabilities = predictor.predict_proba(data)
            predictions = probabilities.argmax(axis=1)

            st.success("Расчет выполнен.")
            st.write("Предсказанные классы:")
            st.write(predictions.tolist())

            st.write("Confidence по классам (0-9):")
            for sample_idx, sample_probs in enumerate(probabilities):
                st.markdown(f"**Образец {sample_idx + 1}**")
                for class_idx, prob in enumerate(sample_probs):
                    color = probability_to_color(prob)
                    percent = prob * 100.0
                    st.markdown(
                        (
                            "<div style='display:flex; align-items:center; gap:10px; margin:4px 0;'>"
                            f"<div style='width:70px;'>Класс {class_idx}</div>"
                            "<div style='flex:1; background:#eee; border-radius:6px; overflow:hidden; height:18px;'>"
                            f"<div style='height:100%; width:{percent:.2f}%; background:{color};'></div>"
                            "</div>"
                            f"<div style='width:64px; text-align:right;'>{percent:.2f}%</div>"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
        except FileNotFoundError:
            st.error(f"Файл весов не найден: {weights_path}")
        except RuntimeError as exc:
            st.error(f"Ошибка при загрузке модели или вычислении: {exc}")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Неожиданная ошибка: {exc}")


if __name__ == "__main__":
    main()
