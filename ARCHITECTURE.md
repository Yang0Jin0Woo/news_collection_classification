# Architecture Note

## Before

The first version was a single Python file. It was easy to run but hard to test.

## After

The expanded version separates responsibilities:

1. Collector: fetches RSS news.
2. Deduplicator: removes duplicate titles by source.
3. Classifier: calls zero-shot classification model.
4. Rule Engine: corrects weak model results with explainable keyword rules.
5. Store: saves results into CSV or SQLite.
6. Report: summarizes final labels and low-confidence cases.
7. CLI/Dashboard: provides different usage interfaces.

## Practical interview point

Do not claim that the core algorithm itself is 10,000 lines. A safe expression is:

> The original core script was about 400 lines. I later expanded it into a modular repository with collectors, classifier adapters, rule engine, storage, reporting, dashboard, and tests. The total repository size can grow close to 10,000 lines when sample cases and tests are included.
