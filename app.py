import streamlit as st
import pandas as pd
import psycopg
from psycopg.rows import dict_row
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import re

# Gemini (optional)
import google.generativeai as genai


# =====================================================
# Config
# =====================================================
APP_TZ = ZoneInfo("Asia/Tokyo")
EXPECTED_COLUMNS = [
    "管理用ID",
    "問題",
    "選択肢１",
    "選択肢２",
    "選択肢３",
    "選択肢４",
    "選択肢５",
    "正答",
    "レベル",
    "解説",
    "主概念",
    "関連概念",
    "要求理解",
    "戻すレベル",
    "戻す概念",
    "簡潔な理由",
    "教員メモ",
]



# =====================================================
# DB (SQLite)
# =====================================================
#@st.cache_resource
def get_conn():
    return psycopg.connect(
        os.environ["DATABASE_URL"],
        row_factory=dict_row
    )


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS question_sets (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TIMESTAMPTZ
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        id SERIAL PRIMARY KEY,
        question_set_id INTEGER REFERENCES question_sets(id) ON DELETE CASCADE,
        qid TEXT,
        question_text TEXT,
        choices_json JSONB,
        correct TEXT,
        level INTEGER,
        primary_concept TEXT,
        related_concepts TEXT,
        required_understanding TEXT,
        fallback_level INTEGER,
        fallback_concept TEXT,
        short_reason TEXT,
        teacher_memo TEXT,
        explanation TEXT,
        UNIQUE(question_set_id, qid)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id SERIAL PRIMARY KEY,
        student_key TEXT UNIQUE
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS answers (
        id SERIAL PRIMARY KEY,
        student_id INTEGER REFERENCES students(id) ON DELETE CASCADE,
        question_id INTEGER REFERENCES questions(id) ON DELETE CASCADE,
        selected TEXT,
        is_correct BOOLEAN,
        answered_at TIMESTAMPTZ,
        coach_json JSONB
    )
    """)

    conn.commit()



def get_or_create_student(student_key: str) -> int:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id FROM students WHERE student_key = %s", (student_key,))
    row = cur.fetchone()
    if row:
        return int(row["id"])

    cur.execute("INSERT INTO students(student_key) VALUES (?)", (student_key,))
    conn.commit()
    return int(cur.lastrowid)


def create_question_set(title: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO question_sets(title, created_at) VALUES (?, ?)",
        (title, datetime.now(APP_TZ).isoformat())
    )
    conn.commit()
    return int(cur.lastrowid)


def upsert_questions(question_set_id: int, df: pd.DataFrame) -> int:
    conn = get_conn()
    cur = conn.cursor()

    inserted = 0
    for _, r in df.iterrows():
        qid = str(r.get("管理用ID", "")).strip()
        qtext = str(r.get("問題", "")).strip()

        choices = {
            "1": str(r.get("選択肢１", "")).strip(),
            "2": str(r.get("選択肢２", "")).strip(),
            "3": str(r.get("選択肢３", "")).strip(),
            "4": str(r.get("選択肢４", "")).strip(),
            "5": str(r.get("選択肢５", "")).strip(),
        }

        correct_raw = str(r.get("正答", "")).strip()
        # 1-5 / A-E どちらでも受ける
        correct_map = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}
        correct = correct_map.get(correct_raw.upper(), correct_raw)
        if correct not in {"1", "2", "3", "4", "5"}:
            correct = "1"

        def to_int(x, default=None):
            try:
                if pd.isna(x):
                    return default
                return int(str(x).strip())
            except Exception:
                return default

        level = to_int(r.get("レベル"), default=4)
        fallback_level = to_int(r.get("戻すレベル"), default=max(1, level - 1))

        explanation = str(r.get("解説", "")).strip()
        primary_concept = str(r.get("主概念", "")).strip()
        related_concepts = str(r.get("関連概念", "")).strip()
        required_understanding = str(r.get("要求理解", "")).strip()
        fallback_concept = str(r.get("戻す概念", "")).strip()
        short_reason = str(r.get("簡潔な理由", "")).strip()
        teacher_memo = str(r.get("教員メモ", "")).strip()

        if not qid:
            qid = f"AUTO_{hash(qtext) & 0xfffffff}"

        if not qtext:
            continue

        cur.execute("""
        INSERT INTO questions (...)
        VALUES (...)
        ON CONFLICT (question_set_id, qid)
        DO UPDATE SET
          question_text = EXCLUDED.question_text,
          choices_json = EXCLUDED.choices_json,
          correct = EXCLUDED.correct;


        # 「解説」はDBに持たせたい場合（任意）：
        # → いまのDBスキーマには explanation列が無いので、
        #   使うならテーブルに列追加が必要です（後述）。

    conn.commit()
    return inserted



def load_questions(question_set_id: int) -> list[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    SELECT * FROM questions
    WHERE question_set_id = ?
    ORDER BY level ASC, id ASC
    """, (question_set_id,))
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def log_answer(student_id: int, question_id: int, selected: str, is_correct: bool, coach_json: dict | None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO answers(student_id, question_id, selected, is_correct, answered_at, coach_json)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        student_id,
        question_id,
        selected,
        1 if is_correct else 0,
        datetime.now(APP_TZ).isoformat(),
        json.dumps(coach_json, ensure_ascii=False) if coach_json else None
    ))
    conn.commit()


def get_student_history(student_id: int, question_set_id: int) -> pd.DataFrame:
    conn = get_conn()
    q = """
    SELECT
        a.id as answer_id,
        a.answered_at,
        a.is_correct,
        a.selected,
        a.coach_json,
        q.level,
        q.primary_concept,
        q.fallback_level,
        q.fallback_concept
    FROM answers a
    JOIN questions q ON a.question_id = q.id
    WHERE a.student_id = ? AND q.question_set_id = ?
    ORDER BY a.id
    """
    return pd.read_sql_query(q, conn, params=(student_id, question_set_id))


# =====================================================
# Gemini coaching (optional)
# =====================================================
def configure_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False
    genai.configure(api_key=api_key)
    return True


def safe_json_extract(text: str) -> dict | None:
    if not text:
        return None
    # remove code fences
    t = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    # find first { ... }
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(t[start:end+1])
    except Exception:
        return None


def ai_coach_diagnose(
    question_text: str,
    choices: dict,
    correct: str,
    selected: str,
    level: int,
    primary_concept: str,
    required_understanding: str,
    fallback_level: int,
    fallback_concept: str,
    short_reason: str
) -> dict:
    """
    Returns dict like:
    {
      "missing_level": 1-3,
      "missing_type": "definition|mechanism|comparison|application",
      "concept": "...",
      "summary": "学生向け1-3文",
      "next_hint": "次に解くべき方向性(短文)"
    }
    """
    if not configure_gemini():
        # Gemini未設定なら、テンプレ診断
        missing_level = max(1, min(3, fallback_level))
        return {
            "missing_level": missing_level,
            "missing_type": "mechanism" if missing_level >= 2 else "definition",
            "concept": fallback_concept or primary_concept or "重要概念",
            "summary": (
                f"今回の誤答は、{required_understanding or '前提理解'}がまだ曖昧な可能性があります。"
                f"まずは「{fallback_concept or primary_concept}」を確認してから、同系統の問題に戻りましょう。"
            ),
            "next_hint": f"戻すレベル {fallback_level} の確認問題へ"
        }

    prompt = f"""
あなたは薬学教育の個別指導コーチです。
目的は叱責や評価ではなく、「なぜ解けなかったか」を理解の階段に沿って言語化し、
次に何を学べばよいかを短く示すことです。

【入力】
- 問題レベル(1-4): {level}
- 主概念: {primary_concept}
- 要求理解: {required_understanding}
- 戻すレベル: {fallback_level}
- 戻す概念: {fallback_concept}

問題文:
{question_text}

選択肢:
{json.dumps(choices, ensure_ascii=False)}

正解: {correct}
学生の選択: {selected}

参考（簡潔な理由）:
{short_reason}

【出力要件（厳守）】
出力はJSONオブジェクト1つのみ。JSON以外の文字は禁止。
キーは以下のみ：
- missing_level（1〜3の整数。推定）
- missing_type（"definition"|"mechanism"|"comparison"|"application"）
- concept（欠けている可能性のある概念。短文）
- summary（学生向け1〜3文。断定禁止、「〜の可能性があります」を用いる。叱責禁止。）
- next_hint（次に取り組むべき方向性を短文で）

補足：
・missing_levelは、戻すレベル/戻す概念の情報も参考にしつつ推定してください。
・暗記ではなく因果や概念のつながりに言及してください。
"""
    model = genai.GenerativeModel(
        "gemini-2.5-flash-lite",
        generation_config={"temperature": 0.2, "max_output_tokens": 450}
    )
    text = model.generate_content(prompt).text.strip()
    data = safe_json_extract(text)
    if not isinstance(data, dict):
        # 失敗時フォールバック
        missing_level = max(1, min(3, fallback_level))
        return {
            "missing_level": missing_level,
            "missing_type": "mechanism" if missing_level >= 2 else "definition",
            "concept": fallback_concept or primary_concept or "重要概念",
            "summary": (
                f"今回の誤答は、{required_understanding or '前提理解'}がまだ曖昧な可能性があります。"
                f"まずは「{fallback_concept or primary_concept}」を確認してから、同系統の問題に戻りましょう。"
            ),
            "next_hint": f"戻すレベル {fallback_level} の確認問題へ"
        }
    return data


# =====================================================
# Adaptive selection
# =====================================================
def pick_next_question(
    questions: list[dict],
    answered_qids: set[int],
    last_result: dict | None,
    last_question: dict | None
) -> dict | None:
    """
    Simple adaptive rule:
    - If last was incorrect: prioritize (fallback_level, fallback_concept) matches
    - If last was correct: try same primary_concept with level+1 if exists else next un-answered
    """
    remaining = [q for q in questions if q["id"] not in answered_qids]
    if not remaining:
        return None

    if last_result and last_question:
        if last_result.get("is_correct") is False:
            target_level = int(last_question.get("fallback_level") or max(1, int(last_question.get("level", 4)) - 1))
            target_concept = (last_question.get("fallback_concept") or "").strip()

            # 1) exact match fallback_level & fallback_concept
            if target_concept:
                cand = [
                    q for q in remaining
                    if int(q.get("level", 4)) == target_level
                    and (q.get("primary_concept") or "").strip() == target_concept
                ]
                if cand:
                    return cand[0]

            # 2) match fallback_level only
            cand = [q for q in remaining if int(q.get("level", 4)) == target_level]
            if cand:
                return cand[0]

            # 3) otherwise pick lowest level remaining
            remaining.sort(key=lambda x: (int(x.get("level", 4)), x["id"]))
            return remaining[0]

        # last correct
        cur_level = int(last_question.get("level", 4))
        up_level = min(4, cur_level + 1)
        cur_concept = (last_question.get("primary_concept") or "").strip()

        # 1) same concept, higher level
        if cur_concept:
            cand = [
                q for q in remaining
                if (q.get("primary_concept") or "").strip() == cur_concept
                and int(q.get("level", 4)) == up_level
            ]
            if cand:
                return cand[0]

    # default: lowest level first
    remaining.sort(key=lambda x: (int(x.get("level", 4)), x["id"]))
    return remaining[0]


def level_label(level: int) -> str:
    return {
        1: "基礎（用語・定義）",
        2: "理由（因果・機序）",
        3: "整理（比較・統合）",
        4: "国家試験レベル（応用）"
    }.get(level, f"レベル{level}")


# =====================================================
# UI
# =====================================================
def main():
    st.set_page_config("段階学習AIコーチ（問題プール版）", layout="centered")
    st.title("📚 段階学習AIコーチ（問題プール選題）")

    init_db()

    # session state
    if "question_set_id" not in st.session_state:
        st.session_state.question_set_id = None
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "current" not in st.session_state:
        st.session_state.current = None
    if "answered_ids" not in st.session_state:
        st.session_state.answered_ids = set()
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_question" not in st.session_state:
        st.session_state.last_question = None

    tab1, tab2, tab3 = st.tabs(["①問題セット取込", "②演習", "③成績・コーチング"])

    with tab1:
        st.subheader("Excelから20問（以上）を取り込む")
        st.write("列名は先生のテンプレートに一致している必要があります。")

        file = st.file_uploader("Excel（.xlsx）またはCSVをアップロード", type=["xlsx", "csv"])
        title = st.text_input("問題セット名（例：薬理1・受容体）", value="My Question Set")

        if file is not None:
            try:
                if file.name.lower().endswith(".xlsx"):
                    df = pd.read_excel(file)
                else:
                    df = pd.read_csv(file)

                missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
                if missing:
                    st.error("列名が不足しています： " + " / ".join(missing))
                    st.stop()

                st.dataframe(df.head(5), use_container_width=True)

                if st.button("このファイルをDBに登録"):
                    qsid = create_question_set(title)
                    count = upsert_questions(qsid, df)
                    st.success(f"登録しました：{count}問")
                    st.session_state.question_set_id = qsid
                    st.session_state.questions = load_questions(qsid)
                    st.session_state.current = None
                    st.session_state.answered_ids = set()
                    st.session_state.last_result = None
                    st.session_state.last_question = None
                    st.rerun()

            except Exception as e:
                st.error("読み込みに失敗しました")
                st.exception(e)

    with tab2:
        st.subheader("問題演習（自動選題）")

        student_key = st.text_input("学籍番号またはニックネーム（必須）", key="student_key")
        if not student_key:
            st.info("学籍番号またはニックネームを入力してください。")
            st.stop()

        if not st.session_state.question_set_id:
            st.info("先に「①問題セット取込」でExcelを登録してください。")
            st.stop()

        student_id = get_or_create_student(student_key)

        # Load questions if needed
        if not st.session_state.questions:
            st.session_state.questions = load_questions(st.session_state.question_set_id)

        # Pick current if none
        if st.session_state.current is None:
            nxt = pick_next_question(
                st.session_state.questions,
                st.session_state.answered_ids,
                st.session_state.last_result,
                st.session_state.last_question
            )
            st.session_state.current = nxt

        # ---------- 問題表示 ----------
        q = st.session_state.current
        if q is None:
            st.success("🎉 すべての問題が終了しました！")
            st.stop()

        level = int(q.get("level", 4))
        st.caption(f"学習段階：{level_label(level)}　/　主概念：{q.get('primary_concept','')}")
        st.markdown("### 問題")
        st.write(q["question_text"])

        # ---------- 選択肢（LaTeX対応） ----------
        choices = json.loads(q["choices_json"])

        st.markdown("### 選択肢")
        for k in ["1", "2", "3", "4", "5"]:
            # LaTeXの$...$が綺麗にレンダリングされる
            st.markdown(f"**{k}.** {choices.get(k,'')}")

        selected = st.radio(
            "解答（番号を選択）",
            options=["1", "2", "3", "4", "5"],
            key=f"choice_{q['id']}"
        )

      # ---------- 解答処理 ----------
        if st.button("解答する"):
            correct = str(q["correct"])
            is_correct = (opt == correct)

            coach = None
            if is_correct:
                coach = {
                    "summary": "正解です。次は同じ概念を少し条件を変えて確認するか、1段階上の問題に進みましょう。",
                    "missing_level": None,
                    "missing_type": None,
                    "concept": q.get("primary_concept", ""),
                    "next_hint": "次の問題へ"
                }
            else:
                coach = ai_coach_diagnose(
                    question_text=q["question_text"],
                    choices=choices,
                    correct=correct,
                    selected=opt,
                    level=level,
                    primary_concept=q.get("primary_concept", ""),
                    required_understanding=q.get("required_understanding", ""),
                    fallback_level=int(q.get("fallback_level") or max(1, level - 1)),
                    fallback_concept=q.get("fallback_concept", ""),
                    short_reason=q.get("short_reason", "")
                )

            log_answer(
                student_id=student_id,
                question_id=int(q["id"]),
                selected=selected,
                is_correct=is_correct,
                coach_json=coach
            )
            
            st.session_state.answered_ids.add(int(q["id"]))
            st.session_state.last_result = {"is_correct": is_correct, "coach": coach}
            st.session_state.last_question = q

            # pick next
            st.session_state.current = pick_next_question(
                st.session_state.questions,
                st.session_state.answered_ids,
                st.session_state.last_result,
                st.session_state.last_question
            )
            
            # show feedback on same run
            if is_correct:
                st.success("正解です 🎉")
            else:
                st.error(f"不正解です。正解は「{correct}」です。")

            st.markdown("### 簡潔な理由")
            st.markdown(q.get("short_reason", "（未記入）"))

            st.markdown("### AIコーチング")
            st.info(coach.get("summary", ""))

            st.divider()
            st.rerun()

    with tab3:
        st.subheader("成績・コーチング（履歴）")

        student_key = st.text_input("学籍番号またはニックネーム", key="student_key_tab3")
        if not student_key:
            st.info("学籍番号またはニックネームを入力してください。")
            st.stop()

        if not st.session_state.question_set_id:
            st.info("問題セットが未登録です。")
            st.stop()

        student_id = get_or_create_student(student_key)
        hist = get_student_history(student_id, st.session_state.question_set_id)

        if hist.empty:
            st.info("履歴がありません。")
            st.stop()

        # 成績サマリ
        st.markdown("### サマリ")
        total = len(hist)
        correct = int(hist["is_correct"].sum())
        st.write(f"正解数：{correct} / {total}（{(correct/total):.0%}）")

        st.markdown("### レベル別 正答率")
        level_stats = hist.groupby("level").agg(
            回答数=("answer_id", "count"),
            正解数=("is_correct", "sum")
        )
        level_stats["正答率"] = level_stats["正解数"] / level_stats["回答数"]
        st.dataframe(level_stats, use_container_width=True)

        st.markdown("### 概念別 正答率")
        concept_stats = hist.groupby("primary_concept").agg(
            回答数=("answer_id", "count"),
            正解数=("is_correct", "sum")
        )
        concept_stats["正答率"] = concept_stats["正解数"] / concept_stats["回答数"]
        st.dataframe(concept_stats.sort_values("正答率"), use_container_width=True)

        st.markdown("### 最近のAIコーチング（最新5件）")
        last5 = hist.tail(5).copy()
        for _, r in last5.iterrows():
            coach = {}
            if r["coach_json"]:
                try:
                    coach = json.loads(r["coach_json"])
                except Exception:
                    coach = {}
            ts = r["answered_at"]
            st.write(f"- {ts} / Level {int(r['level'])} / 概念: {r['primary_concept']} / {'○' if r['is_correct']==1 else '×'}")
            if coach.get("summary"):
                st.info(coach["summary"])

        st.divider()
        if st.button("この学生の進捗をリセット（この問題セットのみ）"):
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("""
            DELETE FROM answers
            WHERE student_id = ?
              AND question_id IN (SELECT id FROM questions WHERE question_set_id = ?)
            """, (student_id, st.session_state.question_set_id))
            conn.commit()
            st.success("リセットしました。")
            st.rerun()


if __name__ == "__main__":
    main()








