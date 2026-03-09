import argparse
import json
import logging
import os
import re

import pandas as pd
from model import Model

LOGGER = logging.getLogger(__name__)

# Set to 0.0 before final submission to disable local validation split.
TRAINING_CONFIG = {
    "validation_split": 0.1,
    "random_state": 42,
    "max_round": 6,
    "selection_metric": "rmse",
    "cv_folds": 5,
    "cv_top_k": 5,
    "cv_prediction_top_k": 5,
    "log_level": "INFO",
    "report_top_k": 3,
}


def get_train_data(input_dir):
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).squeeze("columns")
    return X_train, y_train


def get_test_data(input_dir):
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
    return X_test


def save_predictions(predictions, output_dir, expected_rows):
    if len(predictions) != expected_rows:
        raise ValueError(
            "Prediction length mismatch: "
            f"expected {expected_rows}, got {len(predictions)}"
        )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "y_pred.csv")
    prediction_frame = pd.DataFrame({"age": predictions})
    prediction_frame.to_csv(output_path, index=False)
    return output_path


def configure_logging(level_name):
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def format_candidate_params(result):
    ordered_keys = [
        "alpha",
        "l1_ratio",
        "variance_threshold",
        "n_components",
        "kernel",
        "C",
        "epsilon",
        "gamma",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "max_features",
        "max_samples",
        "num_leaves",
        "base_candidates",
    ]
    values = []
    for key in ordered_keys:
        if key in result and result[key] is not None:
            values.append(f"{key}={result[key]}")
    return ", ".join(values) if values else "default"


def log_tuning_report(model, top_k=3):
    if not model.validation_results_:
        LOGGER.info("Validation split disabled. Training uses full training data directly.")
        return

    LOGGER.info(
        "Validation tuning report (selection_metric=%s, top_k=%d)",
        model.selection_metric,
        top_k,
    )
    successful = [result for result in model.validation_results_ if result.get("status") == "ok"]
    for result in successful[:top_k]:
        LOGGER.info(
            "  rank_%d: %s (%s) [%s], mae=%.4f, rmse=%.4f, custom_score=%.4f, selection_value=%.4f",
            result["rank"],
            result["name"],
            result["regressor"],
            format_candidate_params(result),
            result["mae"],
            result["rmse"],
            result["custom_score"],
            result["selection_value"],
        )

    failed = [result for result in model.validation_results_ if result.get("status") == "failed"]
    if failed:
        LOGGER.warning("Failed candidates (%d):", len(failed))
        for result in failed:
            LOGGER.warning(
                "  - %s (%s): %s",
                result["name"],
                result["regressor"],
                result.get("error", "Unknown failure"),
            )


def run_cross_validation(model, X_train, y_train, cv_folds, cv_top_k):
    if cv_folds is None or cv_folds <= 1:
        LOGGER.info("Cross-validation disabled (cv_folds <= 1).")
        return []

    candidates = model.get_top_successful_candidates(top_k=cv_top_k)
    if not candidates:
        LOGGER.warning("No candidates available for cross-validation.")
        return []

    LOGGER.info(
        "Running cross-validation for top %d candidates with %d folds.",
        len(candidates),
        cv_folds,
    )
    cv_results = model.cross_validate_candidates(
        X=X_train,
        y=y_train,
        candidates=candidates,
        cv_folds=cv_folds,
    )
    successful = [result for result in cv_results if result["status"] == "ok"]
    for result in successful:
        LOGGER.info(
            "  CV rank_%d: %s (%s) rmse=%.4f±%.4f mae=%.4f±%.4f",
            result["cv_rank"],
            result["name"],
            result["regressor"],
            result["rmse_mean"],
            result["rmse_std"],
            result["mae_mean"],
            result["mae_std"],
        )
    failed = [result for result in cv_results if result["status"] == "failed"]
    for result in failed:
        LOGGER.warning(
            "  CV failed: %s (%s): %s",
            result["name"],
            result["regressor"],
            result.get("error", "Unknown error"),
        )
    return cv_results


def sanitize_token(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_")


def save_top_cv_predictions(
    model,
    X_train,
    y_train,
    X_test,
    cv_results,
    output_dir,
    top_k=5,
    test_name=None,
):
    if top_k is None or top_k <= 0:
        LOGGER.info("Top-CV prediction export disabled (top_k <= 0).")
        return None

    successful = [result for result in (cv_results or []) if result.get("status") == "ok"][:top_k]
    if not successful:
        LOGGER.warning("No successful CV candidates available for top prediction export.")
        return None

    effective_test_name = test_name or os.path.basename(os.path.normpath(output_dir)) or "test"
    safe_test_name = sanitize_token(effective_test_name)
    target_folder = os.path.join(
        output_dir,
        f"{safe_test_name}_cv_top{len(successful)}_predictions",
    )
    os.makedirs(target_folder, exist_ok=True)

    cv_rank_by_name = {result["name"]: int(result["cv_rank"]) for result in successful}
    prediction_results = model.predict_candidates_on_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        candidates=successful,
    )

    manifest_rows = []
    failed_exports = []
    for result in prediction_results:
        candidate_name = result["name"]
        candidate_rank = cv_rank_by_name.get(candidate_name, 999)
        candidate_regressor = result.get("regressor", "unknown")

        if result["status"] != "ok":
            failed_exports.append(
                {
                    "name": candidate_name,
                    "regressor": candidate_regressor,
                    "error": result.get("error", "Unknown prediction error"),
                }
            )
            continue

        safe_name = sanitize_token(candidate_name)
        prediction_filename = f"cv_rank_{candidate_rank:02d}_{safe_name}_y_pred.csv"
        prediction_path = os.path.join(target_folder, prediction_filename)
        pd.DataFrame({"age": result["predictions"]}).to_csv(prediction_path, index=False)
        manifest_rows.append(
            {
                "cv_rank": candidate_rank,
                "name": candidate_name,
                "regressor": candidate_regressor,
                "prediction_file": prediction_filename,
            }
        )

    manifest_frame = pd.DataFrame(manifest_rows).sort_values("cv_rank")
    manifest_path = os.path.join(target_folder, "manifest.csv")
    manifest_frame.to_csv(manifest_path, index=False)

    summary = {
        "test_name": effective_test_name,
        "folder": target_folder,
        "manifest": manifest_path,
        "saved_predictions": manifest_rows,
        "failed_predictions": failed_exports,
    }
    LOGGER.info(
        "Saved top CV predictions for test '%s' in %s",
        effective_test_name,
        target_folder,
    )
    if failed_exports:
        LOGGER.warning("Some top CV predictions failed: %s", failed_exports)
    return summary


def save_tuning_report(model, output_dir, cv_results=None, top_predictions_summary=None):
    os.makedirs(output_dir, exist_ok=True)
    report = model.get_tuning_report()
    report_data = report["results"]
    report["cross_validation"] = cv_results or []
    report["top_cv_predictions"] = top_predictions_summary

    if report_data:
        report_frame = pd.DataFrame(report_data)
    else:
        best_model = report.get("best_model") or {}
        report_frame = pd.DataFrame(
            [
                {
                    "rank": 1,
                    **best_model,
                    "mae": None,
                    "rmse": None,
                    "custom_score": None,
                    "note": "Validation disabled (validation_split=0.0).",
                }
            ]
        )

    preferred_columns = [
        "rank",
        "round",
        "name",
        "regressor",
        "status",
        "selection_metric",
        "selection_value",
        "mae",
        "rmse",
        "custom_score",
        "alpha",
        "l1_ratio",
        "variance_threshold",
        "n_components",
        "kernel",
        "C",
        "epsilon",
        "gamma",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "max_features",
        "max_samples",
        "num_leaves",
        "base_candidates",
        "description",
        "usual_use",
        "good_points",
        "bad_points",
        "error",
        "note",
    ]
    table_columns = [column for column in preferred_columns if column in report_frame.columns]
    report_frame_for_export = report_frame[table_columns].copy()

    csv_path = os.path.join(output_dir, "model_tuning_report.csv")
    report_frame_for_export.to_csv(csv_path, index=False)

    cv_csv_path = None
    cv_frame = pd.DataFrame(report["cross_validation"])
    if not cv_frame.empty:
        cv_columns = [
            column
            for column in [
                "cv_rank",
                "name",
                "regressor",
                "status",
                "cv_folds",
                "rmse_mean",
                "rmse_std",
                "mae_mean",
                "mae_std",
                "custom_score_mean",
                "custom_score_std",
                "error",
            ]
            if column in cv_frame.columns
        ]
        cv_frame = cv_frame[cv_columns]
        cv_csv_path = os.path.join(output_dir, "cross_validation_report.csv")
        cv_frame.to_csv(cv_csv_path, index=False)

    json_path = os.path.join(output_dir, "model_tuning_report.json")
    with open(json_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    markdown_path = os.path.join(output_dir, "model_tuning_report.md")
    with open(markdown_path, "w", encoding="utf-8") as report_file:
        report_file.write("# Model Tuning Report\n\n")
        report_file.write(f"- Tuning enabled: `{report['tuning_enabled']}`\n")
        report_file.write(f"- Validation split: `{report['validation_split']}`\n")
        report_file.write(f"- Random state: `{report['random_state']}`\n")
        report_file.write(f"- Max round: `{report['max_round']}`\n")
        report_file.write(f"- Selection metric: `{report['selection_metric']}`\n")
        if report.get("best_model"):
            report_file.write(
                f"- Selected model: `{report['best_model']['name']}`"
                f" (`{report['best_model']['regressor']}`)\n"
            )
        report_file.write("\n## Candidate Results\n\n")
        report_file.write("| " + " | ".join(table_columns) + " |\n")
        report_file.write("| " + " | ".join(["---"] * len(table_columns)) + " |\n")
        for _, row in report_frame_for_export.iterrows():
            values = []
            for column in table_columns:
                value = row[column]
                if isinstance(value, list):
                    value = ", ".join(str(item) for item in value)
                values.append(str(value))
            report_file.write("| " + " | ".join(values) + " |\n")
        report_file.write("\n")

        if not cv_frame.empty:
            report_file.write("## Cross-Validation Results\n\n")
            cv_columns = list(cv_frame.columns)
            report_file.write("| " + " | ".join(cv_columns) + " |\n")
            report_file.write("| " + " | ".join(["---"] * len(cv_columns)) + " |\n")
            for _, row in cv_frame.iterrows():
                values = [str(row[column]) for column in cv_columns]
                report_file.write("| " + " | ".join(values) + " |\n")
            report_file.write("\n")

        if top_predictions_summary:
            report_file.write("## Top Cross-Validation Predictions\n\n")
            report_file.write(f"- Test name: `{top_predictions_summary['test_name']}`\n")
            report_file.write(f"- Folder: `{top_predictions_summary['folder']}`\n")
            report_file.write(f"- Manifest: `{top_predictions_summary['manifest']}`\n")
            report_file.write("\n")
            if top_predictions_summary["saved_predictions"]:
                report_file.write("| cv_rank | name | regressor | prediction_file |\n")
                report_file.write("| --- | --- | --- | --- |\n")
                for row in top_predictions_summary["saved_predictions"]:
                    report_file.write(
                        f"| {row['cv_rank']} | {row['name']} | {row['regressor']} | {row['prediction_file']} |\n"
                    )
                report_file.write("\n")
            if top_predictions_summary["failed_predictions"]:
                report_file.write("Failed prediction exports:\n\n")
                for failed in top_predictions_summary["failed_predictions"]:
                    report_file.write(
                        f"- `{failed['name']}` ({failed['regressor']}): {failed['error']}\n"
                    )
                report_file.write("\n")

    return {
        "csv": csv_path,
        "json": json_path,
        "markdown": markdown_path,
        "cv_csv": cv_csv_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=TRAINING_CONFIG["validation_split"],
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=TRAINING_CONFIG["max_round"],
    )
    parser.add_argument(
        "--selection-metric",
        choices=["rmse", "mae", "custom_score"],
        default=TRAINING_CONFIG["selection_metric"],
    )
    parser.add_argument(
        "--report-top-k",
        type=int,
        default=TRAINING_CONFIG["report_top_k"],
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=TRAINING_CONFIG["cv_folds"],
    )
    parser.add_argument(
        "--cv-top-k",
        type=int,
        default=TRAINING_CONFIG["cv_top_k"],
    )
    parser.add_argument(
        "--cv-pred-top-k",
        type=int,
        default=TRAINING_CONFIG["cv_prediction_top_k"],
    )
    parser.add_argument(
        "--test-name",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    configure_logging(TRAINING_CONFIG["log_level"])

    LOGGER.info("Reading data from %s", input_dir)
    X_train, y_train = get_train_data(input_dir)
    X_test = get_test_data(input_dir)
    LOGGER.info("Loaded train shape=%s and test shape=%s", X_train.shape, X_test.shape)

    model = Model(
        validation_split=args.validation_split,
        random_state=TRAINING_CONFIG["random_state"],
        max_round=args.max_round,
        selection_metric=args.selection_metric,
    )
    LOGGER.info(
        "Training model with validation_split=%.3f max_round=%d selection_metric=%s",
        args.validation_split,
        args.max_round,
        args.selection_metric,
    )
    model.fit(X_train, y_train)
    log_tuning_report(model=model, top_k=args.report_top_k)
    cv_results = run_cross_validation(
        model=model,
        X_train=X_train,
        y_train=y_train,
        cv_folds=args.cv_folds,
        cv_top_k=args.cv_top_k,
    )
    top_predictions_summary = save_top_cv_predictions(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        cv_results=cv_results,
        output_dir=output_dir,
        top_k=args.cv_pred_top_k,
        test_name=args.test_name,
    )
    report_paths = save_tuning_report(
        model=model,
        output_dir=output_dir,
        cv_results=cv_results,
        top_predictions_summary=top_predictions_summary,
    )
    LOGGER.info(
        "Saved tuning reports: csv=%s json=%s markdown=%s cv_csv=%s",
        report_paths["csv"],
        report_paths["json"],
        report_paths["markdown"],
        report_paths["cv_csv"],
    )

    LOGGER.info("Running prediction")
    predictions = model.predict(X_test)
    output_path = save_predictions(
        predictions=predictions,
        output_dir=output_dir,
        expected_rows=len(X_test),
    )
    LOGGER.info("Predictions saved to %s", output_path)


if __name__ == "__main__":
    main()
