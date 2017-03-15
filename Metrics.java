
public class Metrics {

  public static double accuracyScore(Matrix testY, Matrix predictionY) {
    int numRows = testY.entries.length;
    int truePositives = 0;

    for (int i=0; i<numRows; i++) {
      truePositives += (int) testY.entries[i][0] == (int) predictionY.entries[i][0] 
          ? 1 : 0;
    }

    return truePositives / (double) numRows;
  }
}