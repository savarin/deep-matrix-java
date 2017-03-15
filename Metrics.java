
public class Metrics {

  public static int[] predictionCounts(Matrix testY, Matrix predictionY) {
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

    int[] counts = new int[4];
    counts[0] = truePositives;
    counts[1] = falsePositives;
    counts[2] = falseNegatives;
    counts[3] = trueNegatives;

    return counts;
  }

  public static double accuracy(Matrix testY, Matrix predictionY) {
    int[] counts = predictionCounts(testY, predictionY);
    return (counts[0] + counts[3]) / (double) testY.shape()[0];
  }

  public static double precision(Matrix testY, Matrix predictionY) {
    int[] counts = predictionCounts(testY, predictionY);
    return counts[0] / (double) (counts[0] + counts[1]);
  }

  public static double recall(Matrix testY, Matrix predictionY) {
    int[] counts = predictionCounts(testY, predictionY);
    return counts[0] / (double) (counts[0] + counts[2]);
  }
}
