from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
questions = pd.read_csv(os.path.join(BASE_DIR, "questions.csv"))

THRESHOLD = 0.75


# ── Score calculator ─────────────────────────────────────────────
def calculate_score(answers):
    score = 0
    weight_sum = 0

    for qid, ans in answers.items():
        rows = questions[questions["question_id"].astype(str) == str(qid)]
        if rows.empty:
            continue
        row = rows.iloc[0]

        try:
            weight = float(row["weight"])
            positive = int(row["positive_direction"])
        except (ValueError, TypeError):
            continue

        value_map = {
            "Always": 1,
            "Often": 0.75,
            "Sometimes": 0.5,
            "Rarely": 0.25,
            "Never": 0,
            "Yes": 1,
            "No": 0
        }

        value = value_map.get(ans, 0.5)
        if positive == 0:
            value = 1 - value

        score += value * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.5
    return score / weight_sum
# ── Decision tree: pick the next best question ───────────────────
def pick_next_question(answers, current_score):
    answered_ids = set(answers.keys())

    remaining = questions[
        ~questions["question_id"].astype(str).isin(answered_ids)
    ].copy()

    if remaining.empty:
        return None

    num_answered = len(answers)

    # First 3 -- highest weight, one per category
    if num_answered < 3:
        remaining_sorted = remaining.sort_values("weight", ascending=False)
        seen_cats = set()
        for _, row in remaining_sorted.iterrows():
            if row["category"] not in seen_cats:
                seen_cats.add(row["category"])
                return row
        return remaining_sorted.iloc[0]

    # Build category history
    answered_categories = []
    for qid in answered_ids:
        rows = questions[questions["question_id"] == qid]
        if not rows.empty:
            answered_categories.append(rows.iloc[0]["category"])

    category_counts = pd.Series(answered_categories).value_counts().to_dict()

    # Cap each category at 3 questions
    over_asked = {cat for cat, count in category_counts.items() if count >= 3}
    filtered = remaining[~remaining["category"].isin(over_asked)]
    if not filtered.empty:
        remaining = filtered

    # Force category switch if last 2 were the same
    if len(answered_categories) >= 2 and answered_categories[-1] == answered_categories[-2]:
        switched = remaining[remaining["category"] != answered_categories[-1]]
        if not switched.empty:
            remaining = switched

    # High score -- ask red flag questions to stress-test
    if current_score > 0.65:
        red_flags = remaining[remaining["red_flag"] == 1]
        if not red_flags.empty:
            return red_flags.sort_values("weight", ascending=False).iloc[0]

    # Low score -- confirm with high-weight questions
    if current_score < 0.35:
        return remaining.sort_values("weight", ascending=False).iloc[0]

    # Default -- least asked category, highest weight within it
    remaining["category_count"] = remaining["category"].map(
        lambda c: category_counts.get(c, 0)
    )
    remaining = remaining.sort_values(
        ["category_count", "weight"],
        ascending=[True, False]
    )

    return remaining.iloc[0]


# ── Home ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    session.clear()
    return render_template("index.html")


# ── Start ────────────────────────────────────────────────────────
@app.route("/start")
def start():
    session["answers"] = {}
    return redirect(url_for("question"))


# ── Question ─────────────────────────────────────────────────────
@app.route("/question")
def question():
    if "answers" not in session:
        return redirect(url_for("index"))

    answers = session["answers"]
    current_score = calculate_score(answers)

    # Stop conditions
    if len(answers) >= 7:
        if current_score >= 0.75 or current_score <= 0.25:
            return redirect(url_for("result"))

    if len(answers) >= 20:
        return redirect(url_for("result"))

    # Get next question
    q = pick_next_question(answers, current_score)

    if q is None:
        return redirect(url_for("result"))

    # Safety check
    try:
        qid   = str(q["question_id"])
        qtext = str(q["question_text"])
        qcat  = str(q["category"])
    except (KeyError, TypeError):
        return redirect(url_for("result"))

    step  = len(answers) + 1
    total = 20

    if current_score > 0.7:
        flavour = "The signs are looking promising..."
    elif current_score > 0.5:
        flavour = "I sense a meaningful connection..."
    elif current_score > 0.3:
        flavour = "There are some shadows here..."
    else:
        flavour = "I need to understand this better..."

    if qcat == "red_flag":
        options = ["Yes", "No"]
    else:
        options = ["Always", "Often", "Sometimes", "Rarely", "Never"]

    return render_template(
        "question.html",
        question=qtext,
        options=options,
        qid=qid,
        step=step,
        total=total,
        flavour=flavour,
        category=qcat
    )


# ── Answer ───────────────────────────────────────────────────────
@app.route("/answer", methods=["POST"])
def answer():
    if "answers" not in session:
        return redirect(url_for("index"))

    qid = request.form.get("qid")
    ans = request.form.get("answer")

    if not qid or not ans:
        return redirect(url_for("question"))

    answers = session["answers"]
    answers[qid] = ans
    session["answers"] = answers

    prob = calculate_score(answers)

    if prob >= THRESHOLD and len(answers) >= 7:
        return redirect(url_for("result"))

    return redirect(url_for("question"))


# ── Back ─────────────────────────────────────────────────────────
@app.route("/back")
def back():
    if "answers" in session and session["answers"]:
        answers = session["answers"]
        last_key = list(answers.keys())[-1]
        del answers[last_key]
        session["answers"] = answers
    return redirect(url_for("question"))


# ── Result ───────────────────────────────────────────────────────
@app.route("/result")
def result():
    answers = session.get("answers", {})
    prob = calculate_score(answers)
    session.clear()

    confidence = round(prob * 100)

    if prob > 0.85:
        prediction = "Long-term"
        duration = "Very likely to last — 10+ years or a lifetime"
        description = "The patterns in your answers show a deeply healthy and stable bond. Strong communication, mutual respect, and emotional investment are all present."
        work_on = {
            "Keep doing": "Regular quality time together",
            "Nurture": "Emotional vulnerability and openness",
            "Watch out for": "Taking each other for granted over time",
            "Strengthen": "Shared goals and future planning"
        }
    elif prob > 0.65:
        prediction = "Stable"
        duration = "Good foundation — likely 3 to 10 years with effort"
        description = "The relationship has solid foundations but some areas need attention. With the right focus, this bond can grow into something lasting."
        work_on = {
            "Focus on": "Active listening and communication",
            "Build": "Trust through consistent actions",
            "Address": "Unresolved tensions before they grow",
            "Invest in": "Shared experiences and new memories"
        }
    elif prob > 0.45:
        prediction = "At Risk"
        duration = "Could last 1 to 3 years without meaningful change"
        description = "There are clear stress points in this relationship. The bond is present but fragile — both partners need to actively invest to move forward."
        work_on = {
            "Prioritise": "Honest conversations about expectations",
            "Repair": "Broken trust or past conflicts",
            "Learn": "Each other's emotional needs",
            "Consider": "Couples therapy or guided reflection"
        }
    else:
        prediction = "Critical"
        duration = "Unlikely to last beyond 12 months without major change"
        description = "The answers suggest significant disconnection. This does not mean the relationship is over, but both partners need to make real and deliberate changes."
        work_on = {
            "Address urgently": "Core compatibility differences",
            "Rebuild": "Basic communication and respect",
            "Reflect on": "Whether both partners want the same things",
            "Seek": "Professional support or mediation"
        }

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        duration=duration,
        description=description,
        details=work_on
    )


if __name__ == "__main__":
    app.run(debug=True)