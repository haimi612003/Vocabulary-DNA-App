import json
import logging
import os

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "Thiếu GEMINI_API_KEY. Tạo file backend/.env (xem .env.example) và điền API key."
    )

GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dna-vocab-backend")

app = Flask(__name__)
CORS(app)


class ApiError(Exception):
    """Lỗi có mã trạng thái HTTP + message rõ ràng để trả về cho frontend."""

    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@app.errorhandler(ApiError)
def handle_api_error(err):
    logger.error("ApiError %s: %s", err.status_code, err.message)
    return jsonify({"error": err.message}), err.status_code


@app.errorhandler(404)
def handle_not_found(_err):
    return jsonify({"error": "Endpoint không tồn tại."}), 404


@app.errorhandler(405)
def handle_method_not_allowed(_err):
    return jsonify({"error": "Phương thức không được hỗ trợ cho endpoint này."}), 405


def get_json_body():
    body = request.get_json(silent=True)
    if body is None:
        raise ApiError("Body request phải là JSON hợp lệ.", 400)
    return body


def require_fields(body, field_names):
    missing = [name for name in field_names if body.get(name) in (None, "")]
    if missing:
        raise ApiError(f"Thiếu tham số bắt buộc: {', '.join(missing)}", 400)


def call_gemini(system_prompt, user_query, response_schema):
    """Gọi Gemini generateContent với structured output, trả về dict đã parse."""
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
        },
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    try:
        resp = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=60,
        )
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Không kết nối được tới Gemini API: {exc}", 502) from exc

    if resp.status_code != 200:
        try:
            err_body = resp.json()
            gemini_msg = err_body.get("error", {}).get("message", resp.text)
        except ValueError:
            gemini_msg = resp.text

        # 400/401/403/429 đến từ chính Gemini (key sai, quota, request sai) -> forward mã đó.
        # Các lỗi khác (5xx từ Gemini) -> trả 502 cho frontend.
        forward_status = resp.status_code if resp.status_code in (400, 401, 403, 404, 429) else 502
        raise ApiError(f"Gemini API lỗi ({resp.status_code}): {gemini_msg}", forward_status)

    data = resp.json()

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        finish_reason = None
        try:
            finish_reason = data["candidates"][0].get("finishReason")
        except (KeyError, IndexError, TypeError):
            pass
        detail = f" (finishReason: {finish_reason})" if finish_reason else ""
        raise ApiError(f"Gemini trả về phản hồi rỗng hoặc bị chặn{detail}.", 502)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ApiError("Không thể phân tích JSON trả về từ Gemini.", 502) from exc


# ---------------------------------------------------------------------------
# JSON schemas (giữ nguyên cấu trúc responseSchema đang dùng ở frontend cũ)
# ---------------------------------------------------------------------------

EVOLUTION_STEP_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "level": {"type": "NUMBER"},
        "stars": {"type": "NUMBER"},
        "en": {"type": "STRING"},
        "vi": {"type": "STRING"},
        "words_count": {"type": "NUMBER"},
        "evolution_note": {"type": "STRING"},
    },
    "required": ["level", "stars", "en", "vi", "words_count", "evolution_note"],
}

EXERCISE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "id": {"type": "NUMBER"},
        "ref_id": {"type": "NUMBER"},
        "vi_prompt": {"type": "STRING"},
        "evolution_note": {"type": "STRING"},
        "correct_en": {"type": "STRING"},
        "hints": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["id", "ref_id", "vi_prompt", "evolution_note", "correct_en", "hints"],
}

DECODE_DNA_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "word": {"type": "STRING"},
        "evolution_path": {"type": "ARRAY", "items": EVOLUTION_STEP_SCHEMA},
        "exercises": {"type": "ARRAY", "items": EXERCISE_SCHEMA},
        "patterns": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "desc": {"type": "STRING"},
                    "example": {"type": "STRING"},
                },
                "required": ["title", "desc", "example"],
            },
        },
    },
    "required": ["word", "evolution_path", "exercises", "patterns"],
}

EXTRA_DNA_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "extra_evolution_path": {"type": "ARRAY", "items": EVOLUTION_STEP_SCHEMA},
        "extra_exercises": {"type": "ARRAY", "items": EXERCISE_SCHEMA},
    },
    "required": ["extra_evolution_path", "extra_exercises"],
}

READING_PASSAGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "passage": {"type": "STRING"},
        "translation": {"type": "STRING"},
        "highlights": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "word": {"type": "STRING"},
                    "meaning_in_context": {"type": "STRING"},
                },
                "required": ["word", "meaning_in_context"],
            },
        },
    },
    "required": ["title", "passage", "translation", "highlights"],
}

SPEAKING_QUESTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {"question_en": {"type": "STRING"}},
    "required": ["question_en"],
}

SPEAKING_MODEL_ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "model_vi": {"type": "STRING"},
        "model_en": {"type": "STRING"},
        "discovered_words": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "word": {"type": "STRING"},
                    "meaning": {"type": "STRING"},
                },
                "required": ["word", "meaning"],
            },
        },
    },
    "required": ["model_vi", "model_en", "discovered_words"],
}

SPEAKING_CHECK_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_correct": {"type": "BOOLEAN"},
        "suggestion": {"type": "STRING"},
    },
    "required": ["is_correct", "suggestion"],
}

EVALUATE_SENTENCE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "NUMBER"},
        "is_correct": {"type": "BOOLEAN"},
        "feedback": {"type": "STRING"},
        "polish": {"type": "STRING"},
        "native_explanation": {"type": "STRING"},
    },
    "required": ["score", "is_correct", "feedback", "polish", "native_explanation"],
}


# ---------------------------------------------------------------------------
# Routes — mỗi route tương ứng 1 chỗ gọi Gemini cũ ở frontend
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/decode-dna")
def decode_dna():
    body = get_json_body()
    require_fields(body, ["word"])
    word = body["word"].strip().lower()

    system_prompt = f"""Bạn là một nhà khoa học ngôn ngữ học kiêm nhà giáo dục tiếng Anh siêu việt.
Nhiệm vụ của bạn là giải mã chuỗi từ vựng "DNA Từ Vựng" cho một từ khóa chỉ định của học viên.
Hãy trả về một chuỗi cấu trúc JSON thuần túy bằng tiếng Việt và tiếng Anh chi tiết theo cấu trúc bên dưới.

YÊU CẦU ĐẶC BIỆT:
1. "evolution_path": Phải chứa đúng 10 câu ví dụ được thiết kế tăng tiến độ khó đều đặn theo chiều sâu ngữ pháp và chiều dài câu (level từ 1 đến 10 tăng dần).
2. "exercises": Phải chứa đúng 10 câu dịch Việt-Anh tương ứng, có biến hóa nhẹ (tiến hóa nhẹ: thay danh từ, đổi đại từ, thay đổi nhẹ thời thì).
3. Trả về định dạng JSON hoàn hảo trực tiếp, không chứa markdown bên ngoài."""

    user_query = f'Hãy xây dựng DNA học tập toàn diện cho từ tiếng Anh sau: "{word}"'

    result = call_gemini(system_prompt, user_query, DECODE_DNA_SCHEMA)
    return jsonify(result)


@app.post("/api/extra-dna")
def extra_dna():
    body = get_json_body()
    require_fields(body, ["word", "start_level", "count"])
    word = body["word"]
    start_level = int(body["start_level"])
    count = int(body["count"])

    system_prompt = f"""Bạn là chuyên gia ngôn ngữ Anh-Việt cao cấp.
Nhiệm vụ của bạn là tiếp tục phát triển chuỗi DNA từ vựng cho từ "{word}".
Hãy tạo thêm đúng {count} câu ví dụ tiến hóa tiếp theo (Cấp độ từ {start_level} đến {start_level + count - 1}) và đúng {count} bài tập dịch tương ứng có "biến hóa nhẹ".
Trả về định dạng JSON chứa danh sách câu ví dụ bổ sung và bài tập bổ sung như cấu trúc dưới đây."""

    user_query = (
        f'Hãy viết tiếp cấu trúc cho từ khóa "{word}". '
        f"Sinh thêm {count} ví dụ (bắt đầu từ cấp độ {start_level}) và bài tập tương ứng."
    )

    result = call_gemini(system_prompt, user_query, EXTRA_DNA_SCHEMA)
    return jsonify(result)


@app.post("/api/reading-passage")
def reading_passage():
    body = get_json_body()
    require_fields(body, ["theme", "length", "words"])
    theme = body["theme"]
    length = body["length"]
    words = body["words"]
    if not isinstance(words, list) or not words:
        raise ApiError("Trường 'words' phải là danh sách từ và không được rỗng.", 400)

    words_joined = ", ".join(words)

    system_prompt = f"""Bạn là nhà sư phạm tiếng Anh bản ngữ có kỹ năng viết văn bản hấp dẫn.
Nhiệm vụ của bạn là viết một bài đọc ngắn (khoảng {length} chữ) theo chủ đề "{theme}".
BẮT BUỘC phải lồng ghép khéo léo các từ vựng chỉ định sau đây vào văn cảnh một cách tự nhiên nhất: [{words_joined}].
Hãy trả về một định dạng JSON thuần tương chứa các trường: title, passage, translation, highlights."""

    user_query = f'Hãy viết một bài đọc với chủ đề "{theme}" chứa các từ khóa: {words_joined}'

    result = call_gemini(system_prompt, user_query, READING_PASSAGE_SCHEMA)
    return jsonify(result)


@app.post("/api/speaking-question")
def speaking_question():
    body = get_json_body()
    require_fields(body, ["topic"])
    topic = body["topic"]
    studied_words = body.get("studied_words", [])
    studied_joined = ", ".join(studied_words)

    system_prompt = f"""Bạn là giám khảo IELTS chuyên nghiệp.
Nhiệm vụ của bạn là sinh ra duy nhất MỘT CÂU HỎI LUYỆN NÓI tiếng Anh thú vị về chủ đề: "{topic}".
BẮT BUỘC cố gắng lồng ghép 1 từ vựng từ danh sách đã học sau nếu phù hợp: [{studied_joined}].
Trả về JSON thuần tuý:
{{
  "question_en": "Câu hỏi bằng tiếng Anh"
}}"""

    user_query = f'Sinh một câu hỏi luyện nói chủ đề "{topic}".'

    result = call_gemini(system_prompt, user_query, SPEAKING_QUESTION_SCHEMA)
    return jsonify(result)


@app.post("/api/speaking-model-answer")
def speaking_model_answer():
    body = get_json_body()
    require_fields(body, ["question", "level"])
    question = body["question"]
    level = body["level"]

    system_prompt = f"""Bạn là giám khảo IELTS chuyên nghiệp.
Nhiệm vụ của bạn là dựa trên câu hỏi sau: "{question}", hãy sinh một câu trả lời mẫu tiếng Việt và tiếng Anh chuẩn mực ở trình độ CEFR: "{level}".
Bắt buộc đề xuất 1-2 từ vựng đắt giá (discovered_words) trong câu trả lời đó.
Cấu trúc trả về là JSON:
{{
  "model_vi": "Một câu trả lời mẫu bằng tiếng Việt",
  "model_en": "Bản dịch tiếng Anh của câu trả lời mẫu ở trình độ {level}",
  "discovered_words": [
    {{ "word": "từ mới", "meaning": "giải nghĩa" }}
  ]
}}"""

    user_query = f'Sinh câu trả lời cho câu hỏi: "{question}" ở trình độ {level}'

    result = call_gemini(system_prompt, user_query, SPEAKING_MODEL_ANSWER_SCHEMA)
    return jsonify(result)


@app.post("/api/speaking-check")
def speaking_check():
    body = get_json_body()
    require_fields(body, ["model_answer", "user_answer"])
    model_answer = body["model_answer"]
    user_answer = body["user_answer"]

    system_prompt = """Bạn là trợ lý chấm điểm phản xạ dịch nói tiếng Anh IELTS.
Nhiệm vụ của bạn là kiểm tra xem câu viết tiếng Anh của người dùng có khớp nghĩa và đúng ngữ pháp so với đáp án mẫu tiếng Anh hay không.
Trả về JSON thuần túy:
{
  "is_correct": true hoặc false,
  "suggestion": "Nhận xét ngắn bằng tiếng Việt."
}"""

    user_query = f'Bản dịch mẫu: "{model_answer}"\nCâu của học viên: "{user_answer}"'

    result = call_gemini(system_prompt, user_query, SPEAKING_CHECK_SCHEMA)
    return jsonify(result)


@app.post("/api/evaluate-sentence")
def evaluate_sentence():
    body = get_json_body()
    require_fields(body, ["word", "sentence"])
    word = body["word"]
    sentence = body["sentence"]

    system_prompt = f"""Bạn là trợ lý giảng dạy tiếng Anh bản xứ.
Nhiệm vụ của bạn là đánh giá câu viết tiếng Anh của học viên có sử dụng từ khóa: "{word}".
Hãy trả về JSON:
{{
  "score": 85,
  "is_correct": true,
  "feedback": "Nhận xét ngắn bằng tiếng Việt",
  "polish": "Câu chuẩn bản ngữ tối ưu",
  "native_explanation": "Giải thích chi tiết"
}}"""

    user_query = f'Hãy kiểm tra câu này: "{sentence}" sử dụng từ khóa "{word}"'

    result = call_gemini(system_prompt, user_query, EVALUATE_SENTENCE_SCHEMA)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
