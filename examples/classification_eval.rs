//! Complete classification evaluation workflow:
//! confusion matrix -> per-class metrics -> ROC/PR curves -> calibration check -> system comparison

use statskit::{
    Average, accuracy, average_precision, balanced_accuracy, brier_score, cohen_kappa,
    expected_calibration_error, f1, hamming_loss, jaccard_score, log_loss,
    maximum_calibration_error, mcc, mcnemar, precision, recall, roc_auc, specificity,
};

fn main() {
    // Ground truth and predictions for binary classification
    let y_true = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0];
    let y_pred = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0];
    let y_prob = [0.9, 0.8, 0.4, 0.75, 0.1, 0.2, 0.6, 0.15, 0.85, 0.3];

    // -- Classification metrics --
    println!("=== Classification Metrics ===");
    println!("Accuracy:          {:.4}", accuracy(&y_true, &y_pred));
    println!(
        "Balanced accuracy: {:.4}",
        balanced_accuracy(&y_true, &y_pred)
    );
    println!(
        "Precision (macro): {:.4}",
        precision(&y_true, &y_pred, Average::Macro)
    );
    println!(
        "Recall (macro):    {:.4}",
        recall(&y_true, &y_pred, Average::Macro)
    );
    println!(
        "F1 (macro):        {:.4}",
        f1(&y_true, &y_pred, Average::Macro)
    );
    println!("MCC:               {:.4}", mcc(&y_true, &y_pred));
    println!("Specificity:       {:.4}", specificity(&y_true, &y_pred));
    println!("Cohen's kappa:     {:.4}", cohen_kappa(&y_true, &y_pred));
    println!("Hamming loss:      {:.4}", hamming_loss(&y_true, &y_pred));
    println!(
        "Jaccard (macro):   {:.4}",
        jaccard_score(&y_true, &y_pred, Average::Macro)
    );

    // -- Probabilistic metrics --
    println!("\n=== Probabilistic Metrics ===");
    println!("Log loss:          {:.4}", log_loss(&y_true, &y_prob));
    println!("ROC AUC:           {:.4}", roc_auc(&y_true, &y_prob));
    println!(
        "Average precision: {:.4}",
        average_precision(&y_true, &y_prob)
    );

    // -- Calibration --
    println!("\n=== Calibration ===");
    println!("Brier score:       {:.4}", brier_score(&y_true, &y_prob));
    println!(
        "ECE (10 bins):     {:.4}",
        expected_calibration_error(&y_true, &y_prob, 10)
    );
    println!(
        "MCE (10 bins):     {:.4}",
        maximum_calibration_error(&y_true, &y_prob, 10)
    );

    // -- System comparison with McNemar's test --
    let pred_b = [1, 0, 0, 0, 0, 0, 0, 0, 1, 1]; // a weaker classifier
    println!("\n=== System Comparison (McNemar) ===");
    let result = mcnemar(&y_true, &y_pred, &pred_b, 0.05);
    println!(
        "b={}, c={}, chi2={:.4}, p={:.4}, significant={}",
        result.b, result.c, result.statistic, result.p_value, result.significant
    );
}
