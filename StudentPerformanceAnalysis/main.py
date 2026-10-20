

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


NAME_COLUMN = "Прізвище та ініціали студентів"
DATE_FORMAT = "%d.%m.%Y"
LOW_GRADE_LIMIT = 7.0
OUTPUT_COLUMNS = [
    "Студент",
    "Середній бал",
    "Кількість оцінок",
    "Bad grades (1–3)",
    "Статус",
]
StudentRecord = tuple[str, float | None, int, int]


def parse_arguments() -> argparse.Namespace:
    """Parse optional paths and student-search settings."""
    parser = argparse.ArgumentParser(
        description="Очищення та аналіз електронного журналу успішності."
    )
    parser.add_argument(
        "--journal",
        type=Path,
        default=Path(__file__).resolve().parent / "journal.csv",
        help="Шлях до CSV-журналу (за замовчуванням: journal.csv поруч зі скриптом).",
    )
    parser.add_argument(
        "--student",
        help="Знайти студента за повним ім’ям без інтерактивного запиту.",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Не запускати пошук студента.",
    )
    return parser.parse_args()


def read_journal(path: Path) -> pd.DataFrame:
    """Read the exported journal, whose fifth row contains column headers."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Не знайдено CSV-журнал: {path}. Додайте файл journal.csv до папки проєкту "
            "або передайте шлях через --journal."
        )
    return pd.read_csv(path, encoding="utf-8-sig", header=4, dtype="string")


def select_student_rows(journal: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Keep named students and convert dated grade cells to numeric values."""
    if NAME_COLUMN not in journal.columns:
        raise ValueError(
            f"У журналі відсутня колонка з іменами студентів: {NAME_COLUMN!r}."
        )

    names = journal[NAME_COLUMN].fillna("").str.strip()
    student_mask = names.ne("")
    students = journal.loc[student_mask].copy()
    names = names.loc[student_mask]
    if students.empty:
        raise ValueError("У журналі не знайдено жодного рядка з ПІБ студента.")

    date_columns: list[tuple[pd.Timestamp, str]] = []
    for column in journal.columns:
        try:
            date = pd.to_datetime(str(column).strip(), format=DATE_FORMAT)
        except (TypeError, ValueError):
            continue
        date_columns.append((date, column))

    if not date_columns:
        raise ValueError("У журналі не знайдено колонок із датами занять.")

    date_columns.sort(key=lambda item: item[0])
    grade_cells = students[[column for _, column in date_columns]].copy()
    grade_cells.columns = [date for date, _ in date_columns]
    numeric_grades = grade_cells.apply(
        lambda column: pd.to_numeric(
            column.str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )
    )
    numeric_grades.index = names.index
    names.name = NAME_COLUMN
    return names, numeric_grades


def insertion_sort_students(records: list[StudentRecord]) -> list[StudentRecord]:
    """Sort records by average ascending, placing students without grades last."""
    sorted_records = records.copy()

    def sort_key(record: StudentRecord) -> tuple[bool, float]:
        average = record[1]
        return (average is None, average if average is not None else 0.0)

    for index in range(1, len(sorted_records)):
        current = sorted_records[index]
        current_key = sort_key(current)
        position = index - 1
        while position >= 0 and sort_key(sorted_records[position]) > current_key:
            sorted_records[position + 1] = sorted_records[position]
            position -= 1
        sorted_records[position + 1] = current

    return sorted_records


def make_student_records(names: pd.Series, grades: pd.DataFrame) -> list[StudentRecord]:
    """Build one summary record per student without replacing missing grades."""
    averages = grades.mean(axis=1)
    grade_counts = grades.count(axis=1)
    low_grade_counts = (grades.ge(1) & grades.le(3)).sum(axis=1)

    records: list[StudentRecord] = []
    for index, name in names.items():
        average_value = averages.loc[index]
        average = None if pd.isna(average_value) else float(average_value)
        records.append(
            (
                name,
                average,
                int(grade_counts.loc[index]),
                int(low_grade_counts.loc[index]),
            )
        )
    return records


def status_for_average(average: float | None) -> str:
    if average is None:
        return "Немає оцінок"
    if average < LOW_GRADE_LIMIT:
        return "Потребує уваги"
    return "Добре"


def write_results(records: list[StudentRecord], output_path: Path) -> pd.DataFrame:
    """Save student summaries in ascending average order."""
    rows = [
        {
            OUTPUT_COLUMNS[0]: name,
            OUTPUT_COLUMNS[1]: round(average, 2) if average is not None else None,
            OUTPUT_COLUMNS[2]: grade_count,
            OUTPUT_COLUMNS[3]: low_grade_count,
            OUTPUT_COLUMNS[4]: status_for_average(average),
        }
        for name, average, grade_count, low_grade_count in records
    ]
    results = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    return results


def print_statistics(grades: pd.DataFrame, records: list[StudentRecord]) -> None:
    values = grades.to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("У датованих колонках журналу немає числових оцінок.")

    averages = [record[1] for record in records if record[1] is not None]
    low_average_count = sum(average < LOW_GRADE_LIMIT for average in averages)
    low_grade_count = int(np.count_nonzero((values >= 1) & (values <= 3)))

    print("\nСтатистика успішності:")
    print(f"Кількість студентів: {len(records)}")
    print(f"Кількість числових оцінок: {values.size}")
    print(f"Середній бал групи: {np.mean(values):.2f}")
    print(f"Найвища оцінка: {np.max(values):g}")
    print(f"Найнижча оцінка: {np.min(values):g}")
    print(f"Кількість оцінок 1–3: {low_grade_count}")
    print(f"Студентів із середнім балом нижче 7: {low_average_count}")

    grade_values, grade_frequencies = np.unique(values, return_counts=True)
    most_common_count = int(grade_frequencies.max())
    most_common_grades = grade_values[grade_frequencies == most_common_count]
    common_text = ", ".join(f"{grade:g}" for grade in most_common_grades)
    print(
        f"Найчастіші оцінки: {common_text} "
        f"(по {most_common_count} разів)"
    )

    dated_averages = grades.mean(axis=0).dropna()
    if not dated_averages.empty:
        lowest_date = dated_averages.idxmin()
        print(
            "Найнижчий середній бал за заняття: "
            f"{lowest_date:%d.%m.%Y} ({dated_averages.loc[lowest_date]:.2f})"
        )


def save_charts(
    grades: pd.DataFrame,
    records: list[StudentRecord],
    output_directory: Path,
) -> None:
    """Create the average, dated trend, and grade distribution charts."""
    scored_records = [record for record in records if record[1] is not None]
    names = [record[0] for record in scored_records]
    averages = np.array([record[1] for record in scored_records], dtype=float)
    colors = ["crimson" if average < LOW_GRADE_LIMIT else "steelblue" for average in averages]

    figure_width = max(12, len(names) * 0.45)
    figure, axis = plt.subplots(figsize=(figure_width, 7))
    axis.bar(names, averages, color=colors)
    axis.axhline(
        LOW_GRADE_LIMIT,
        color="darkred",
        linestyle="--",
        linewidth=1,
        label="Поріг уваги (7 балів)",
    )
    axis.set_title("Середній бал студентів")
    axis.set_xlabel("Студенти")
    axis.set_ylabel("Середній бал")
    axis.tick_params(axis="x", labelrotation=75)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "average_grades.png", dpi=150)
    plt.close(figure)

    dated_averages = grades.mean(axis=0).dropna()
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(
        dated_averages.index,
        dated_averages.to_numpy(),
        marker="o",
        color="darkcyan",
    )
    axis.set_title("Динаміка успішності групи")
    axis.set_xlabel("Дата заняття")
    axis.set_ylabel("Середній бал")
    axis.tick_params(axis="x", labelrotation=45)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_directory / "performance_trend.png", dpi=150)
    plt.close(figure)

    numeric_values = grades.to_numpy(dtype=float).ravel()
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    lower_edge = math.floor(float(numeric_values.min())) - 0.5
    upper_edge = math.ceil(float(numeric_values.max())) + 1.5
    bins = np.arange(lower_edge, upper_edge + 1, 1)

    figure, axis = plt.subplots(figsize=(10, 6))
    _, bin_edges, patches = axis.hist(
        numeric_values,
        bins=bins,
        edgecolor="white",
    )
    for left_edge, right_edge, patch in zip(bin_edges[:-1], bin_edges[1:], patches):
        grade = (left_edge + right_edge) / 2
        if 1 <= grade <= 3:
            patch.set_facecolor("crimson")
        else:
            patch.set_facecolor("steelblue")
    axis.set_title("Розподіл числових оцінок")
    axis.set_xlabel("Оцінка")
    axis.set_ylabel("Кількість")
    axis.set_xticks(np.arange(math.ceil(lower_edge), math.floor(upper_edge) + 1))
    axis.legend(
        handles=[
            Patch(facecolor="crimson", label="Початковий рівень (1–3)"),
            Patch(facecolor="steelblue", label="Інші оцінки"),
        ]
    )
    figure.tight_layout()
    figure.savefig(output_directory / "grades_distribution.png", dpi=150)
    plt.close(figure)


def find_student(records: list[StudentRecord], query: str) -> StudentRecord | None:
    """Find a student by a case-insensitive full-name comparison."""
    normalized_query = query.strip().casefold()
    if not normalized_query:
        return None
    for record in records:
        if record[0].strip().casefold() == normalized_query:
            return record
    return None


def run_student_search(records: list[StudentRecord], query: str) -> None:
    student = find_student(records, query)
    if student is None:
        print(f"Студента «{query.strip()}» не знайдено.")
        return

    name, average, grade_count, _ = student
    average_text = f"{average:.2f}" if average is not None else "немає оцінок"
    print(f"\nРезультат пошуку: {name}")
    print(f"Середній бал: {average_text}")
    print(f"Кількість оцінок: {grade_count}")
    print(f"Статус: {status_for_average(average)}")


def main() -> int:
    arguments = parse_arguments()
    journal = read_journal(arguments.journal)

    print("Перші 5 рядків журналу:")
    print(journal.head().to_string(max_cols=12))
    print(f"\nРозмір початкової таблиці (рядки, колонки): {journal.shape}")
    print(f"Назви колонок ({len(journal.columns)}): {journal.columns.tolist()}")

    names, grades = select_student_rows(journal)
    records = make_student_records(names, grades)
    insertion_sorted = insertion_sort_students(records)
    reference_sorted = sorted(
        records,
        key=lambda record: (
            record[1] is None,
            record[1] if record[1] is not None else 0.0,
        ),
    )
    if insertion_sorted != reference_sorted:
        raise AssertionError("Власне сортування не збігається з sorted().")
    print("\nПеревірка сортування вставками та sorted(): збігаються.")

    output_directory = arguments.journal.resolve().parent
    results_path = output_directory / "results.csv"
    results = write_results(insertion_sorted, results_path)
    print("\nП’ять найнижчих середніх балів:")
    print(results.head(5).to_string(index=False))

    print_statistics(grades, records)
    save_charts(grades, records, output_directory)
    print(f"\nЗбережено таблицю результатів: {results_path}")
    print(f"Збережено графіки в папці: {output_directory}")

    if not arguments.skip_search:
        if arguments.student is not None:
            run_student_search(records, arguments.student)
        else:
            try:
                query = input("\nВведіть повне ім’я студента для пошуку: ")
            except EOFError:
                print("Пошук пропущено: введення недоступне. Запустіть із --skip-search.")
            else:
                if query.strip():
                    run_student_search(records, query)
                else:
                    print("Пошук пропущено: ім’я не введене.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())