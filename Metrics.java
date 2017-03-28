
public class Metrics {

  /**
   * Counts of predictions for model performance metrics.
   *
   * @param testY Test labels matrix.
   * @param predictionY Predicted labels matrix.
   * @return int[] Prediction counts.
   */
  public static int[] counts(Matrix testY, Matrix predictionY) {
    int numRows = testY.shape()[0];
    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;
    int trueNegatives = 0;

    for (int i = 0; i < numRows; i++) {
      truePositives +=
          (int) testY.entries[i][0] == 1 && (int) predictionY.entries[i][0] == 1 ? 1 : 0;
      falsePositives +=
          (int) testY.entries[i][0] == 0 && (int) predictionY.entries[i][0] == 1 ? 1 : 0;
      falseNegatives +=
          (int) testY.entries[i][0] == 1 && (int) predictionY.entries[i][0] == 0 ? 1 : 0;
      trueNegatives +=
          (int) testY.entries[i][0] == 0 && (int) predictionY.entries[i][0] == 0 ? 1 : 0;
    }

    int[] predictionCounts = new int[4];
    predictionCounts[0] = truePositives;
    predictionCounts[1] = falsePositives;
    predictionCounts[2] = falseNegatives;
    predictionCounts[3] = trueNegatives;

    return predictionCounts;
  }

  /**
   * Accuracy measures the proportion of correct predictions made.
   *
   * @param testY Test labels matrix.
   * @param predictionY Predicted labels matrix.
   * @return double Accuracy score.
   */
  public static double accuracy(Matrix testY, Matrix predictionY) {
    int[] predictionCounts = counts(testY, predictionY);
    return (predictionCounts[0] + predictionCounts[3]) / (double) testY.shape()[0];
  }

  /**
   * Precision measures the proportion of correctly identified positives over all positive
   * predictions made.
   *
   * @param testY Test labels matrix.
   * @param predictionY Predicted labels matrix.
   * @return double Precision score.
   */
  public static double precision(Matrix testY, Matrix predictionY) {
    int[] predictionCounts = counts(testY, predictionY);
    return predictionCounts[0] / (double) (predictionCounts[0] + predictionCounts[1]);
  }

  /**
   * Recall measures the proportion of positives that were correctly identified.
   *
   * @param testY Test labels matrix.
   * @param predictionY Predicted labels matrix.
   * @return Recall score.
   */
  public static double recall(Matrix testY, Matrix predictionY) {
    int[] predictionCounts = counts(testY, predictionY);
    return predictionCounts[0] / (double) (predictionCounts[0] + predictionCounts[2]);
  }
}
