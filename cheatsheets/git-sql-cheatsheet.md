# 🔧 Git & SQL Cheatsheet

Two skills every AI engineer needs daily — version control and querying data.

---

## Git — Daily Commands

```bash
git init                          # start a repo
git clone <url>                   # copy a remote repo
git status                        # see what changed
git add file.py                   # stage a file
git add .                         # stage everything
git commit -m "message"           # save a snapshot
git push origin main               # upload to GitHub
git pull origin main               # download latest changes
```

## Git — Branching

```bash
git branch feature-x              # create branch
git checkout feature-x            # switch to it
git checkout -b feature-x         # create + switch in one step
git merge feature-x               # merge into current branch
git branch -d feature-x           # delete branch after merge
```

## Git — Undoing Things

```bash
git checkout -- file.py           # discard unstaged changes
git reset HEAD file.py            # unstage a file
git reset --soft HEAD~1           # undo last commit, keep changes
git reset --hard HEAD~1           # undo last commit, discard changes ⚠️
git revert <commit-hash>          # safely undo a pushed commit
```

## Git — Inspecting

```bash
git log --oneline                 # compact commit history
git diff                          # see unstaged changes
git diff --staged                 # see staged changes
git stash                         # temporarily save changes
git stash pop                     # bring them back
```

## Good Commit Messages

```
feat: add logistic regression model training script
fix: handle missing values in salary column
docs: update README with Phase 4 progress
refactor: extract data cleaning into separate function
```

---

## SQL — Core Queries

```sql
SELECT column1, column2 FROM table_name WHERE condition;

SELECT * FROM employees WHERE salary > 50000 ORDER BY salary DESC;

SELECT DISTINCT department FROM employees;

SELECT * FROM employees LIMIT 10;
```

## SQL — Aggregation

```sql
SELECT department, AVG(salary) AS avg_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 60000
ORDER BY avg_salary DESC;

SELECT COUNT(*), MAX(salary), MIN(salary), SUM(salary) FROM employees;
```

## SQL — Joins

```sql
-- INNER JOIN: only matching rows
SELECT e.name, d.dept_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.id;

-- LEFT JOIN: all from left, matched from right
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id;
```

| Join Type | Returns |
|---|---|
| INNER JOIN | Only matching rows in both tables |
| LEFT JOIN | All rows from left + matches from right |
| RIGHT JOIN | All rows from right + matches from left |
| FULL OUTER JOIN | All rows from both, matched where possible |

## SQL — Window Functions

```sql
SELECT name, salary,
       RANK() OVER (ORDER BY salary DESC) AS salary_rank,
       AVG(salary) OVER (PARTITION BY department) AS dept_avg
FROM employees;
```

## SQL — CTEs (Common Table Expressions)

```sql
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 80000
)
SELECT department, COUNT(*) FROM high_earners GROUP BY department;
```

## SQL Quick Reference

| Clause Order (write) | Execution Order |
|---|---|
| SELECT | FROM |
| FROM | WHERE |
| WHERE | GROUP BY |
| GROUP BY | HAVING |
| HAVING | SELECT |
| ORDER BY | ORDER BY |
| LIMIT | LIMIT |
