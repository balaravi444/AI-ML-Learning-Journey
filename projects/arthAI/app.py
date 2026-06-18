"""
ArthAI — FastAPI Web Application
Smart Financial Advisor for Every Indian 🇮🇳

Author: Bala Ravi
Date: 17 June 2026
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from modules.goal_planner import generate_goal_report
from pydantic import BaseModel
from modules.financial_utils import (
    calculate_emi,
    total_interest_paid,
    future_value_sip,
    find_minimum_sip,
    smart_budget_plan,
    calculate_tax_savings,
    retirement_corpus_needed,
    loan_amortization
)

app = FastAPI(title="ArthAI", version="1.0.0")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ===== Request Models =====
class BudgetRequest(BaseModel):
    monthly_income: float


class EMIRequest(BaseModel):
    principal: float
    rate: float
    years: int


class SIPRequest(BaseModel):
    target: float
    rate: float
    years: int


class RetirementRequest(BaseModel):
    current_age: int
    retirement_age: int
    monthly_expenses: float


class TaxRequest(BaseModel):
    annual_income: float
    investment_80c: float = 0
    health_insurance: float = 0
    nps: float = 0
class Goal(BaseModel):
    name: str
    target_amount: float
    years: int
    priority: int


class GoalPlannerRequest(BaseModel):
    goals: list[Goal]
    monthly_savings: float


# ===== Routes =====
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render main ArthAI dashboard."""
    return templates.TemplateResponse(
        "index.html", {"request": request})


@app.post("/api/budget")
async def get_budget_plan(data: BudgetRequest):
    """Calculate smart budget plan."""
    plan = smart_budget_plan(data.monthly_income)
    return plan


@app.post("/api/emi")
async def get_emi_analysis(data: EMIRequest):
    """Calculate EMI and loan analysis."""
    analysis = total_interest_paid(
        data.principal, data.rate, data.years)
    schedule = loan_amortization(
        data.principal, data.rate, data.years)
    return {
        "analysis": analysis,
        "schedule_preview": schedule[:6]
    }


@app.post("/api/sip")
async def get_sip_plan(data: SIPRequest):
    """Calculate SIP investment plan."""
    min_sip = find_minimum_sip(
        data.target, data.rate, data.years)
    corpus = future_value_sip(
        min_sip, data.rate, data.years)

    comparisons = []
    for multiplier in [1, 1.5, 2]:
        sip_amt = int(min_sip * multiplier)
        c = future_value_sip(sip_amt, data.rate, data.years)
        comparisons.append({
            "sip": sip_amt,
            "corpus": round(c)
        })

    return {
        "min_sip": min_sip,
        "expected_corpus": round(corpus),
        "comparisons": comparisons
    }


@app.post("/api/retirement")
async def get_retirement_plan(data: RetirementRequest):
    """Calculate retirement planning."""
    plan = retirement_corpus_needed(
        data.current_age,
        data.retirement_age,
        data.monthly_expenses
    )
    return plan


@app.post("/api/tax")
async def get_tax_savings(data: TaxRequest):
    """Calculate tax savings."""
    investments = {
        "80c_investments": data.investment_80c,
        "health_insurance": data.health_insurance,
        "nps": data.nps
    }
    result = calculate_tax_savings(
        data.annual_income, investments)
    return result
    
@app.post("/api/goals")
async def get_goal_plan(data: GoalPlannerRequest):
    """Calculate optimized multi-goal plan."""
    from modules.financial_utils import find_minimum_sip

    goals_with_amounts = []
    for g in data.goals:
        monthly = find_minimum_sip(g.target_amount, 10, g.years)
        goals_with_amounts.append({
            "name": g.name,
            "target_amount": g.target_amount,
            "years": g.years,
            "priority": g.priority,
            "monthly_required": monthly
        })

    report = generate_goal_report(
        goals_with_amounts, data.monthly_savings)
    return report



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
