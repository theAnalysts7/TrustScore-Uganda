from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pytesseract
from PIL import Image, ImageEnhance, ImageOps
import re, threading, math

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///trustscore.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'trustscore-uganda-2026'
db = SQLAlchemy(app)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ══════════════════ MODELS ══════════════════
class User(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    phone         = db.Column(db.String(20),  unique=True, nullable=False)
    password      = db.Column(db.String(200), nullable=False)
    provider      = db.Column(db.String(20),  default='MTN')
    created_at    = db.Column(db.String(30),  default=str(datetime.now()))

class Transaction(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, nullable=False)
    date          = db.Column(db.String(20))
    type          = db.Column(db.String(10))
    amount        = db.Column(db.Float)
    description   = db.Column(db.String(200))
    source        = db.Column(db.String(10), default='csv')
    screenshot_no = db.Column(db.Integer,    default=1)
    uploaded_at   = db.Column(db.String(30), default=str(datetime.now()))

class Score(db.Model):
    id                  = db.Column(db.Integer, primary_key=True)
    user_id             = db.Column(db.Integer, nullable=False)
    score               = db.Column(db.Float)
    total_income        = db.Column(db.Float)
    total_spending      = db.Column(db.Float)
    net_savings         = db.Column(db.Float)
    total_transactions  = db.Column(db.Integer)
    total_credits       = db.Column(db.Integer, default=0)
    total_debits        = db.Column(db.Integer, default=0)
    avg_credit_amount   = db.Column(db.Float,   default=0)
    avg_debit_amount    = db.Column(db.Float,   default=0)
    screenshots_used    = db.Column(db.Integer, default=1)
    loan_recommended    = db.Column(db.Float,   default=0)
    computed_at         = db.Column(db.String(30), default=str(datetime.now()))

# ══════════════════ OCR ENGINE ══════════════════
def preprocess_image(img):
    """Return 3 preprocessed versions of the image for best OCR accuracy."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w, h = img.size
    if w < 900:
        scale = 900 / w
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    gray = img.convert('L')
    versions = []
    # 1. Plain black & white
    versions.append(gray.point(lambda x: 255 if x > 140 else 0, '1').convert('RGB'))
    # 2. Contrast-enhanced BW
    enh = ImageEnhance.Contrast(img).enhance(2.0)
    enh = ImageEnhance.Sharpness(enh).enhance(1.8)
    versions.append(enh.convert('L').point(lambda x: 255 if x > 128 else 0, '1').convert('RGB'))
    # 3. Inverted BW (good for dark-background screenshots)
    versions.append(ImageOps.invert(gray).point(lambda x: 255 if x > 128 else 0, '1').convert('RGB'))
    return versions

def ocr_image(image_file):
    """Run OCR on all 3 image versions in parallel; return the one with most text."""
    try:
        img = Image.open(image_file)
        versions = preprocess_image(img)
        results  = ['' for _ in versions]
        def worker(i, v):
            try:
                results[i] = pytesseract.image_to_string(v, config='--psm 6 --oem 3')
            except Exception as e:
                print(f'[OCR worker {i}] {e}')
        threads = [threading.Thread(target=worker, args=(i, v)) for i, v in enumerate(versions)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=30)
        best = max(results, key=lambda s: len(s.strip()))
        print(f'[OCR] chars={len(best)} preview={best[:80].replace(chr(10)," ")}')
        return best
    except Exception as e:
        print(f'[OCR ERROR] {e}')
        return ''

# ══════════════════ PATTERN ENGINE ══════════════════
MTN_CREDIT = [
    r'you\s+have\s+received\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'received\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'ugx?\s*([\d,]+(?:\.\d+)?)\s+(?:has been )?(?:credited|deposited|received)',
    r'cash\s+in\s+of\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'salary\s+(?:of\s+)?ugx?\s*([\d,]+(?:\.\d+)?)',
    r'payment\s+received\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'has\s+sent\s+you\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'momo\s+pay\s+received\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'confirmed\.\s+ugx?\s*([\d,]+(?:\.\d+)?)\s+received',
    r'deposit(?:ed)?\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'reversal\s+of\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'transfer\s+received\s+ugx?\s*([\d,]+(?:\.\d+)?)',
]
MTN_DEBIT = [
    r'you\s+have\s+(?:sent|paid|transferred)\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'sent\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'paid\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'ugx?\s*([\d,]+(?:\.\d+)?)\s+(?:has been )?(?:debited|deducted|withdrawn|sent)',
    r'cash\s+out\s+of\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'withdrawal\s+of\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'payment\s+of\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'you\s+have\s+bought\s+airtime\s+of\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'charge[sd]?\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'fee[sd]?\s+(?:of\s+)?ugx?\s*([\d,]+(?:\.\d+)?)',
]
AIRTEL_CREDIT = [
    r'you\s+have\s+received\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'received\s+ugx?\s*([\d,]+(?:\.\d+)?)\s+from',
    r'ugx?\s*([\d,]+(?:\.\d+)?)\s+(?:has been )?deposited',
    r'money\s+in[:\s]+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'credit[ed]?\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'refund\s+of\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'ugx?\s*([\d,]+(?:\.\d+)?)\s+credited',
]
AIRTEL_DEBIT = [
    r'you\s+have\s+sent\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'sent\s+ugx?\s*([\d,]+(?:\.\d+)?)\s+to',
    r'ugx?\s*([\d,]+(?:\.\d+)?)\s+(?:has been )?(?:debited|deducted|withdrawn)',
    r'money\s+out[:\s]+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'charged\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'deducted\s+ugx?\s*([\d,]+(?:\.\d+)?)',
    r'transfer[red]?\s+ugx?\s*([\d,]+(?:\.\d+)?)\s+to',
]
AMOUNT_FALLBACK = re.compile(r'(?:ugx?|shs?)\.?\s*([\d,]{3,}(?:\.\d{1,2})?)', re.IGNORECASE)
DATE_RE = re.compile(
    r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})'
    r'|(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})',
    re.IGNORECASE
)
BALANCE_RE = re.compile(
    r'(?:new\s+)?(?:balance|bal)[\s:]+(?:ugx?|shs?)?\s*([\d,]+(?:\.\d+)?)',
    re.IGNORECASE
)

def clean_amount(s):
    try:
        v = float(s.replace(',', '').strip())
        return v if 200 <= v <= 100_000_000 else None
    except:
        return None

def parse_screenshot(text):
    """
    Parse a single screenshot's OCR text.
    Returns (provider, list_of_transactions, list_of_balances).
    Each transaction: {date, type, amount, description}
    """
    tl = text.lower()
    lines = text.split('\n')

    # Detect provider
    if any(k in tl for k in ['mtn', 'momo', 'mobile money']):
        provider, cp, dp = 'MTN', MTN_CREDIT, MTN_DEBIT
    elif 'airtel' in tl:
        provider, cp, dp = 'Airtel', AIRTEL_CREDIT, AIRTEL_DEBIT
    else:
        # Default — try MTN patterns first
        provider, cp, dp = 'MTN', MTN_CREDIT, MTN_DEBIT

    txns = []

    def find_txn(text_block, patterns, txn_type):
        for p in patterns:
            m = re.search(p, text_block, re.IGNORECASE)
            if m:
                amt = clean_amount(m.group(1))
                if amt:
                    return amt
        return None

    # Strategy 1: Block-by-block (each SMS message is usually separated by blank lines)
    blocks = re.split(r'\n{2,}', text)
    for block in blocks:
        bl = block.lower()
        if not bl.strip() or len(bl) < 10:
            continue
        amt = find_txn(bl, cp, 'credit')
        if amt:
            dm = DATE_RE.search(bl)
            txns.append({
                'date': dm.group(0) if dm else datetime.now().strftime('%Y-%m-%d'),
                'type': 'credit', 'amount': amt,
                'description': block.strip()[:150].replace('\n', ' ')
            })
            continue
        amt = find_txn(bl, dp, 'debit')
        if amt:
            dm = DATE_RE.search(bl)
            txns.append({
                'date': dm.group(0) if dm else datetime.now().strftime('%Y-%m-%d'),
                'type': 'debit', 'amount': amt,
                'description': block.strip()[:150].replace('\n', ' ')
            })

    # Strategy 2: Line-by-line with 3-line context window
    for i, line in enumerate(lines):
        ctx = ' '.join(lines[max(0,i-1): min(len(lines),i+3)]).lower()
        # Skip if this line's content is already captured
        if any(t['description'] and line.strip()[:25].lower() in t['description'].lower() for t in txns):
            continue
        amt = find_txn(ctx, cp, 'credit')
        if amt and not any(abs(t['amount']-amt)<1 and t['type']=='credit' for t in txns):
            dm = DATE_RE.search(ctx)
            txns.append({
                'date': dm.group(0) if dm else datetime.now().strftime('%Y-%m-%d'),
                'type': 'credit', 'amount': amt,
                'description': line.strip()[:150]
            })
            continue
        amt = find_txn(ctx, dp, 'debit')
        if amt and not any(abs(t['amount']-amt)<1 and t['type']=='debit' for t in txns):
            dm = DATE_RE.search(ctx)
            txns.append({
                'date': dm.group(0) if dm else datetime.now().strftime('%Y-%m-%d'),
                'type': 'debit', 'amount': amt,
                'description': line.strip()[:150]
            })

    # Strategy 3: Generic UGX fallback if nothing found
    if not txns:
        credit_kw = ['received','credit','deposit','salary','refund','reversal','in']
        debit_kw  = ['sent','paid','debit','withdraw','charge','bought','out','transfer','fee']
        for i, line in enumerate(lines):
            ctx = ' '.join(lines[max(0,i-1): min(len(lines),i+3)]).lower()
            for match in AMOUNT_FALLBACK.findall(ctx):
                amt = clean_amount(match)
                if not amt:
                    continue
                txn_type = None
                for w in credit_kw:
                    if w in ctx: txn_type = 'credit'; break
                if not txn_type:
                    for w in debit_kw:
                        if w in ctx: txn_type = 'debit'; break
                if txn_type and not any(abs(t['amount']-amt)<1 for t in txns):
                    dm = DATE_RE.search(ctx)
                    txns.append({
                        'date': dm.group(0) if dm else datetime.now().strftime('%Y-%m-%d'),
                        'type': txn_type, 'amount': amt,
                        'description': line.strip()[:150]
                    })

    # Extract balance snapshots
    balances = []
    for m in BALANCE_RE.finditer(text):
        try:
            b = float(m.group(1).replace(',',''))
            if 0 < b < 500_000_000:
                balances.append(b)
        except:
            pass

    print(f'[PARSE] provider={provider} txns={len(txns)} balances={len(balances)}')
    return provider, txns, balances


# ══════════════════ TRUST SCORE ENGINE ══════════════════
def calculate_trust_score(all_txns, all_balances, num_screenshots):
    """
    Calculate a 0–100 trust score from the combined transaction list.

    Key insight: num_screenshots tells us how many data points (screenshots)
    were uploaded. We use this to estimate per-screenshot averages.
    Each screenshot represents ONE batch of transactions (not necessarily one month).

    Returns: (score, breakdown_dict, total_income, total_spending,
              n_txns, n_credits, n_debits, avg_credit, avg_debit)
    """
    if not all_txns:
        return 0.0, {}, 0.0, 0.0, 0, 0, 0, 0.0, 0.0

    df = pd.DataFrame(all_txns)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df.dropna(subset=['amount'], inplace=True)
    if df.empty:
        return 0.0, {}, 0.0, 0.0, 0, 0, 0, 0.0, 0.0

    credits_df = df[df['type'] == 'credit']
    debits_df  = df[df['type'] == 'debit']

    total_income   = float(credits_df['amount'].sum())
    total_spending = float(debits_df['amount'].sum())
    n_txns         = len(df)
    n_credits      = len(credits_df)
    n_debits       = len(debits_df)
    avg_credit     = float(credits_df['amount'].mean()) if n_credits > 0 else 0.0
    avg_debit      = float(debits_df['amount'].mean())  if n_debits  > 0 else 0.0

    if total_income == 0:
        bd = {
            'income_stability': 0, 'spending_discipline': 0,
            'transaction_activity': min(100, n_txns * 4),
            'savings_behavior': 0, 'balance_trend': 50, 'repayment_capacity': 0
        }
        return 5.0, bd, total_income, total_spending, n_txns, n_credits, n_debits, avg_credit, avg_debit

    # ── Factor 1: Income Stability (25%) ──
    # Measures how consistent income amounts are
    std_credit = float(credits_df['amount'].std()) if n_credits > 1 else 0.0
    cv = std_credit / avg_credit if avg_credit > 0 else 1.0
    income_stability = max(0.0, min(100.0, 100.0 - cv * 55.0))
    # Reward more credit transactions (more evidence = more trust)
    income_stability = min(100.0, income_stability + min(15.0, n_credits * 1.5))
    # Reward multiple screenshots
    income_stability = min(100.0, income_stability + (num_screenshots - 1) * 2.0)

    # ── Factor 2: Spending Discipline (25%) ──
    spending_ratio = total_spending / total_income
    if   spending_ratio <= 0.25: sd = 100.0
    elif spending_ratio <= 0.40: sd = 90.0
    elif spending_ratio <= 0.55: sd = 75.0
    elif spending_ratio <= 0.70: sd = 58.0
    elif spending_ratio <= 0.85: sd = 38.0
    elif spending_ratio <= 1.00: sd = 18.0
    else:                        sd = max(0.0, 18.0 - (spending_ratio - 1.0) * 18.0)

    # ── Factor 3: Transaction Activity (15%) ──
    activity = min(100.0, n_txns * 3.5)
    if n_credits > 0 and n_debits > 0:
        activity = min(100.0, activity + 5.0)  # Having both types is healthy

    # ── Factor 4: Savings Behavior (20%) ──
    savings_ratio = (total_income - total_spending) / total_income
    if   savings_ratio >= 0.50: sav = 100.0
    elif savings_ratio >= 0.35: sav = 85.0
    elif savings_ratio >= 0.20: sav = 68.0
    elif savings_ratio >= 0.10: sav = 50.0
    elif savings_ratio >= 0.00: sav = 28.0
    else:                       sav = max(0.0, 28.0 + savings_ratio * 35.0)

    # ── Factor 5: Balance Trend (10%) ──
    bt = 65.0
    if all_balances and len(all_balances) >= 2:
        mid = len(all_balances) // 2
        first_half  = sum(all_balances[:mid]) / mid
        second_half = sum(all_balances[mid:]) / (len(all_balances) - mid)
        if second_half > first_half * 1.15:   bt = 92.0
        elif second_half > first_half:          bt = 78.0
        elif second_half > first_half * 0.90:  bt = 62.0
        else:                                   bt = 38.0
    else:
        # Use income trend as proxy
        credit_amounts = credits_df['amount'].tolist()
        if len(credit_amounts) >= 4:
            mid = len(credit_amounts) // 2
            bt = 80.0 if sum(credit_amounts[mid:]) >= sum(credit_amounts[:mid]) else 50.0

    # ── Factor 6: Repayment Capacity (5%) ──
    net_per_screenshot = (total_income - total_spending) / max(num_screenshots, 1)
    income_per_screenshot = total_income / max(num_screenshots, 1)
    rep_cap = min(100.0, max(0.0, net_per_screenshot / income_per_screenshot * 100)) if income_per_screenshot > 0 else 0.0

    # ── Final weighted score ──
    final = (
        income_stability * 0.25 +
        sd               * 0.25 +
        activity         * 0.15 +
        sav              * 0.20 +
        bt               * 0.10 +
        rep_cap          * 0.05
    )

    breakdown = {
        'income_stability':     round(income_stability, 1),
        'spending_discipline':  round(sd, 1),
        'transaction_activity': round(activity, 1),
        'savings_behavior':     round(sav, 1),
        'balance_trend':        round(bt, 1),
        'repayment_capacity':   round(rep_cap, 1),
    }

    return (round(min(100.0, max(0.0, final)), 2),
            breakdown, total_income, total_spending,
            n_txns, n_credits, n_debits, avg_credit, avg_debit)


# ══════════════════ LOAN PREDICTION ENGINE ══════════════════
def predict_loan(score, total_income, total_spending, num_screenshots,
                 n_credits, avg_credit, breakdown):
    """
    Predict loan amount from ACTUAL extracted transaction data.

    Logic:
    - avg_income_per_screenshot = total_income / num_screenshots
      (Each screenshot is one data batch; income per batch = proxy for periodic income)
    - disposable = avg_income_per_screenshot - avg_spending_per_screenshot
    - max repayment = 35% of disposable (conservative, investor-safe)
    - Loan = present value of annuity: amount you can borrow given that repayment
    - Capped by tier limits and income multipliers
    """

    if num_screenshots == 0:
        num_screenshots = 1

    avg_income_per_shot  = total_income  / num_screenshots
    avg_spending_per_shot = total_spending / num_screenshots
    disposable_per_shot  = max(0.0, avg_income_per_shot - avg_spending_per_shot)

    # Repayment capacity: 35% of disposable per period
    max_monthly_repayment = disposable_per_shot * 0.35

    # Determine tier
    if score >= 80:
        tier = 'Gold';   duration = 18; rate = 0.18; multiplier = 3.5
        tier_cap   = 15_000_000; tier_floor = 500_000
    elif score >= 65:
        tier = 'Silver'; duration = 12; rate = 0.22; multiplier = 2.5
        tier_cap   = 6_000_000;  tier_floor = 200_000
    elif score >= 50:
        tier = 'Bronze'; duration = 6;  rate = 0.28; multiplier = 1.5
        tier_cap   = 2_500_000;  tier_floor = 50_000
    else:
        # Not eligible — still explain why based on real numbers
        needed = round(max(0, 50 - score), 1)
        return {
            'eligible': False, 'tier': 'None',
            'max_loan': 0, 'max_loan_fmt': 'UGX 0',
            'monthly_payment': 0, 'monthly_payment_fmt': 'N/A',
            'duration_months': 0, 'duration': 'N/A',
            'interest_rate': 0,   'interest_fmt': 'N/A',
            'total_repayment': 0, 'total_repayment_fmt': 'N/A',
            'total_interest': 0,  'total_interest_fmt': 'N/A',
            'avg_income_per_shot': round(avg_income_per_shot),
            'avg_income_per_shot_fmt': ugx(avg_income_per_shot),
            'avg_spending_per_shot': round(avg_spending_per_shot),
            'avg_spending_per_shot_fmt': ugx(avg_spending_per_shot),
            'disposable_per_shot': round(disposable_per_shot),
            'disposable_per_shot_fmt': ugx(disposable_per_shot),
            'max_monthly_repayment': round(max_monthly_repayment),
            'max_monthly_repayment_fmt': ugx(max_monthly_repayment),
            'income_multiplier': 0, 'dti_ratio': 0,
            'screenshots_used': num_screenshots,
            'improvement_needed': needed,
            'message': (
                f'Your score of {score}% is below the minimum 50% threshold. '
                f'You need {needed} more points. '
                f'Based on your {num_screenshots} screenshot(s), your average income per batch is '
                f'{ugx(avg_income_per_shot)} with spending of {ugx(avg_spending_per_shot)}. '
                f'Reduce spending and save more to qualify.'
            )
        }

    # ── Annuity formula: loan principal from periodic payment ──
    # P = R * [(1 - (1+i)^-n) / i]
    monthly_rate = rate / 12
    if monthly_rate > 0 and max_monthly_repayment > 0:
        annuity_factor = (1 - (1 + monthly_rate) ** (-duration)) / monthly_rate
        loan_from_repayment = max_monthly_repayment * annuity_factor
    else:
        loan_from_repayment = max_monthly_repayment * duration

    # ── Income cap: can't borrow more than X times your periodic income ──
    income_cap = avg_income_per_shot * multiplier

    # ── Final loan amount ──
    loan_amount = min(loan_from_repayment, income_cap, tier_cap)

    # Enforce floor only if user actually has income
    if avg_income_per_shot > 0:
        loan_amount = max(loan_amount, tier_floor)
    else:
        loan_amount = 0

    # Round to nearest UGX 10,000
    loan_amount = math.floor(loan_amount / 10_000) * 10_000

    # ── Actual monthly payment for that loan ──
    if monthly_rate > 0 and loan_amount > 0:
        monthly_payment = loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** (-duration))
    else:
        monthly_payment = 0

    total_repayment = monthly_payment * duration
    total_interest  = total_repayment - loan_amount
    dti = round(avg_spending_per_shot / avg_income_per_shot * 100, 1) if avg_income_per_shot > 0 else 0

    msgs = {
        'Gold':   (f'Outstanding profile! Your {num_screenshots} screenshots show average income of '
                   f'{ugx(avg_income_per_shot)} per batch with {ugx(disposable_per_shot)} disposable. '
                   f'You qualify for our highest loan tier.'),
        'Silver': (f'Strong financial profile. Your {num_screenshots} screenshots verify average income of '
                   f'{ugx(avg_income_per_shot)} per batch. You qualify for a Silver tier loan.'),
        'Bronze': (f'Fair profile. Your {num_screenshots} screenshots show income of '
                   f'{ugx(avg_income_per_shot)} per batch. You qualify for a starter loan. '
                   f'Upload more screenshots and improve savings to unlock higher amounts.'),
    }

    return {
        'eligible': True, 'tier': tier,
        'max_loan': int(loan_amount), 'max_loan_fmt': ugx(loan_amount),
        'monthly_payment': int(monthly_payment), 'monthly_payment_fmt': ugx(monthly_payment),
        'duration_months': duration, 'duration': f'{duration} months',
        'interest_rate': rate, 'interest_fmt': f'{int(rate*100)}% p.a.',
        'total_repayment': int(total_repayment), 'total_repayment_fmt': ugx(total_repayment),
        'total_interest':  int(total_interest),  'total_interest_fmt':  ugx(total_interest),
        'avg_income_per_shot': int(avg_income_per_shot),
        'avg_income_per_shot_fmt': ugx(avg_income_per_shot),
        'avg_spending_per_shot': int(avg_spending_per_shot),
        'avg_spending_per_shot_fmt': ugx(avg_spending_per_shot),
        'disposable_per_shot': int(disposable_per_shot),
        'disposable_per_shot_fmt': ugx(disposable_per_shot),
        'max_monthly_repayment': int(max_monthly_repayment),
        'max_monthly_repayment_fmt': ugx(max_monthly_repayment),
        'income_multiplier': multiplier, 'dti_ratio': dti,
        'screenshots_used': num_screenshots,
        'message': msgs[tier],
    }


def ugx(n):
    return f"UGX {int(max(0, n)):,}"


def get_tips(score, breakdown, loan):
    tips = []
    bd = breakdown

    if bd.get('spending_discipline', 100) < 60:
        tips.append({"icon":"💸","title":"Reduce Your Spending",
            "body":"Your spending is above 70% of your income. Getting it below 55% will move you to the next loan tier."})
    if bd.get('income_stability', 100) < 55:
        tips.append({"icon":"📈","title":"Stabilise Your Income",
            "body":"Irregular income reduces trust. Consistent deposits — even small ones — significantly improve your score."})
    if bd.get('savings_behavior', 100) < 50:
        tips.append({"icon":"🏦","title":"Save More Regularly",
            "body":"Your savings rate is below 20%. Saving 20–35% of income each period unlocks Gold tier loans."})
    if bd.get('transaction_activity', 100) < 40:
        tips.append({"icon":"🔄","title":"Transact More Often",
            "body":"Low transaction count reduces your activity score. Use mobile money regularly to build a stronger profile."})
    if score >= 80:
        tips.append({"icon":"⭐","title":"Excellent — Keep It Up",
            "body":"Your financial behaviour is outstanding. Maintain this to keep Gold tier access and grow your loan limit over time."})
    elif score >= 65:
        diff = round(80 - score, 1)
        tips.append({"icon":"🎯","title":f"Only {diff}pts from Gold",
            "body":"Reduce spending and increase savings this period to break into Gold tier and unlock significantly higher loans."})
    if loan.get('eligible') and loan.get('dti_ratio', 0) > 65:
        tips.append({"icon":"⚠️","title":"High Debt-to-Income Ratio",
            "body":f"Your DTI is {loan['dti_ratio']}%. Responsible lenders prefer under 50%. Reducing outflows will improve your loan terms."})
    return tips[:4]


# ══════════════════ ROUTES ══════════════════
@app.route('/')
def home():
    return "TrustScore API v3 ✅"

@app.route('/init-db')
def init_db():
    db.create_all()
    return "Database initialised successfully!"

@app.route('/register', methods=['POST'])
def register():
    d = request.json
    if not d or not all(k in d for k in ['name','phone','password','provider']):
        return jsonify({"error":"All fields required"}), 400
    if User.query.filter_by(phone=d['phone']).first():
        return jsonify({"error":"Phone number already registered"}), 409
    u = User(name=d['name'], phone=d['phone'],
             password=generate_password_hash(d['password']), provider=d['provider'])
    db.session.add(u); db.session.commit()
    return jsonify({"message":"Registration successful","user_id":u.id,"name":u.name})

@app.route('/login', methods=['POST'])
def login():
    d = request.json
    if not d or not all(k in d for k in ['phone','password']):
        return jsonify({"error":"Phone and password required"}), 400
    u = User.query.filter_by(phone=d['phone']).first()
    if not u or not check_password_hash(u.password, d['password']):
        return jsonify({"error":"Invalid phone or password"}), 401
    return jsonify({"message":"Login successful","user_id":u.id,"name":u.name,"provider":u.provider})


# ── MULTI-SCREENSHOT UPLOAD (main endpoint) ──
@app.route('/upload-screenshots/<int:user_id>', methods=['POST'])
def upload_screenshots(user_id):
    files = request.files.getlist('screenshots')
    if not files:
        return jsonify({"error":"No screenshots uploaded"}), 400
    if len(files) < 2:
        return jsonify({"error":
            "Please upload at least 2 screenshots for a meaningful analysis. "
            "5 screenshots give the highest loan amount."}), 400

    all_txns    = []
    all_balances = []
    per_shot    = []
    providers   = []

    for i, f in enumerate(files):
        ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
        if ext not in {'png','jpg','jpeg','bmp','webp','gif','tiff'}:
            per_shot.append({"index":i+1,"filename":f.filename,"status":"skipped",
                             "error":"Invalid image format","txns":0,"income":0,"spending":0})
            continue

        raw = ocr_image(f)
        if not raw or len(raw.strip()) < 5:
            per_shot.append({"index":i+1,"filename":f.filename,"status":"failed",
                             "error":"Could not read text from image","txns":0,"income":0,"spending":0})
            continue

        provider, txns, bals = parse_screenshot(raw)
        providers.append(provider)
        all_txns.extend(txns)
        all_balances.extend(bals)

        # Per-screenshot stats for the UI
        shot_income   = sum(t['amount'] for t in txns if t['type']=='credit')
        shot_spending = sum(t['amount'] for t in txns if t['type']=='debit')
        per_shot.append({
            "index": i+1, "filename": f.filename, "status":"ok",
            "provider": provider, "txns": len(txns),
            "income": round(shot_income), "income_fmt": ugx(shot_income),
            "spending": round(shot_spending), "spending_fmt": ugx(shot_spending),
        })

    if not all_txns:
        return jsonify({
            "error": "No transactions could be extracted from any screenshot. "
                     "Ensure screenshots clearly show MTN/Airtel transaction messages.",
            "per_screenshot": per_shot
        }), 400

    num_ok = sum(1 for s in per_shot if s['status'] == 'ok')

    (score, breakdown,
     total_income, total_spending,
     n_txns, n_credits, n_debits,
     avg_credit, avg_debit) = calculate_trust_score(all_txns, all_balances, num_ok)

    loan = predict_loan(score, total_income, total_spending,
                        num_ok, n_credits, avg_credit, breakdown)
    tips = get_tips(score, breakdown, loan)

    # Persist transactions
    for idx, txn in enumerate(all_txns):
        db.session.add(Transaction(
            user_id=user_id,
            date=str(txn.get('date','')),
            type=txn['type'],
            amount=float(txn['amount']),
            description=str(txn.get('description',''))[:200],
            source='image',
            screenshot_no=1  # we flatten across all screenshots
        ))

    db.session.add(Score(
        user_id=user_id, score=float(score),
        total_income=total_income, total_spending=total_spending,
        net_savings=total_income-total_spending,
        total_transactions=n_txns,
        total_credits=n_credits, total_debits=n_debits,
        avg_credit_amount=avg_credit, avg_debit_amount=avg_debit,
        screenshots_used=num_ok,
        loan_recommended=float(loan.get('max_loan', 0))
    ))
    db.session.commit()

    detected_provider = max(set(providers), key=providers.count) if providers else 'Unknown'
    return jsonify({
        "source": "screenshots",
        "provider_detected": detected_provider,
        "screenshots_processed": num_ok,
        "screenshots_uploaded": len(files),
        "per_screenshot": per_shot,
        "total_transactions": n_txns,
        "total_credits": n_credits,
        "total_debits": n_debits,
        "total_income": round(total_income, 2),
        "total_income_fmt": ugx(total_income),
        "total_spending": round(total_spending, 2),
        "total_spending_fmt": ugx(total_spending),
        "net_savings": round(total_income - total_spending, 2),
        "net_savings_fmt": ugx(total_income - total_spending),
        "avg_credit_amount": round(avg_credit, 2),
        "avg_debit_amount":  round(avg_debit, 2),
        "trust_score": float(score),
        "breakdown": breakdown,
        "loan_recommendation": loan,
        "tips": tips,
    })


# ── CSV UPLOAD ──
@app.route('/upload/<int:user_id>', methods=['POST'])
def upload_csv(user_id):
    if 'file' not in request.files:
        return jsonify({"error":"No file uploaded"}), 400
    try:
        df = pd.read_csv(request.files['file'])
    except Exception:
        return jsonify({"error":"Invalid CSV file"}), 400

    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df.dropna(subset=['amount'], inplace=True)
    if 'type' not in df.columns:
        return jsonify({"error":"CSV must have a 'type' column with values 'credit' or 'debit'"}), 400

    txns = df.to_dict('records')
    (score, breakdown,
     total_income, total_spending,
     n_txns, n_credits, n_debits,
     avg_credit, avg_debit) = calculate_trust_score(txns, [], 1)

    loan = predict_loan(score, total_income, total_spending,
                        1, n_credits, avg_credit, breakdown)
    tips = get_tips(score, breakdown, loan)

    for row in txns:
        db.session.add(Transaction(
            user_id=user_id, date=str(row.get('date','')),
            type=row['type'], amount=float(row['amount']),
            description='CSV import', source='csv'
        ))
    db.session.add(Score(
        user_id=user_id, score=float(score),
        total_income=total_income, total_spending=total_spending,
        net_savings=total_income-total_spending,
        total_transactions=n_txns,
        total_credits=n_credits, total_debits=n_debits,
        avg_credit_amount=avg_credit, avg_debit_amount=avg_debit,
        screenshots_used=1, loan_recommended=float(loan.get('max_loan',0))
    ))
    db.session.commit()

    return jsonify({
        "source": "csv",
        "total_transactions": n_txns,
        "total_credits": n_credits, "total_debits": n_debits,
        "total_income": round(total_income,2), "total_income_fmt": ugx(total_income),
        "total_spending": round(total_spending,2), "total_spending_fmt": ugx(total_spending),
        "net_savings": round(total_income-total_spending,2),
        "net_savings_fmt": ugx(total_income-total_spending),
        "avg_credit_amount": round(avg_credit,2),
        "avg_debit_amount": round(avg_debit,2),
        "trust_score": float(score),
        "breakdown": breakdown,
        "loan_recommendation": loan,
        "tips": tips,
        "screenshots_processed": 1,
    })


@app.route('/dashboard/<int:user_id>', methods=['GET'])
def dashboard(user_id):
    u  = User.query.get(user_id)
    if not u:
        return jsonify({"error":"User not found"}), 404
    ls = Score.query.filter_by(user_id=user_id).order_by(Score.id.desc()).first()
    all_scores = Score.query.filter_by(user_id=user_id).order_by(Score.id.asc()).all()
    recent_txns = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.id.desc()).limit(5).all()
    total_txns  = Transaction.query.filter_by(user_id=user_id).count()
    total_cr    = db.session.query(db.func.sum(Transaction.amount)).filter_by(user_id=user_id, type='credit').scalar() or 0
    total_db    = db.session.query(db.func.sum(Transaction.amount)).filter_by(user_id=user_id, type='debit').scalar() or 0
    trend = [{"score":s.score, "date":s.computed_at[:10] if s.computed_at else '',
              "loan":s.loan_recommended} for s in all_scores[-10:]]
    return jsonify({
        "user": {"name":u.name,"phone":u.phone,"provider":u.provider,
                 "member_since":u.created_at[:10] if u.created_at else ''},
        "latest_score": {
            "score":ls.score, "total_income":ls.total_income,
            "total_spending":ls.total_spending, "net_savings":ls.net_savings,
            "total_transactions":ls.total_transactions,
            "total_credits":ls.total_credits, "total_debits":ls.total_debits,
            "avg_credit_amount":ls.avg_credit_amount, "avg_debit_amount":ls.avg_debit_amount,
            "screenshots_used":ls.screenshots_used, "loan_recommended":ls.loan_recommended,
            "computed_at":ls.computed_at
        } if ls else None,
        "score_trend": trend,
        "summary": {
            "total_transactions":total_txns,
            "total_income":float(total_cr), "total_spending":float(total_db),
            "total_scans":len(all_scores)
        },
        "recent_transactions": [
            {"date":t.date,"type":t.type,"amount":t.amount,
             "description":t.description,"source":t.source} for t in recent_txns
        ],
    })


@app.route('/history/<int:user_id>', methods=['GET'])
def history(user_id):
    ss = Score.query.filter_by(user_id=user_id).order_by(Score.id.desc()).all()
    return jsonify({"history": [{
        "score":s.score, "total_income":s.total_income,
        "total_spending":s.total_spending, "net_savings":s.net_savings,
        "total_transactions":s.total_transactions,
        "avg_credit_amount":s.avg_credit_amount, "avg_debit_amount":s.avg_debit_amount,
        "screenshots_used":s.screenshots_used, "loan_recommended":s.loan_recommended,
        "computed_at":s.computed_at
    } for s in ss]})


@app.route('/transactions/<int:user_id>', methods=['GET'])
def transactions(user_id):
    tt = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.id.desc()).limit(60).all()
    return jsonify({"transactions": [
        {"date":t.date,"type":t.type,"amount":t.amount,
         "description":t.description,"source":t.source} for t in tt
    ]})


if __name__ == '__main__':
    app.run(debug=True)