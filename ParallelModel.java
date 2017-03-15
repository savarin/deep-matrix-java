
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Represents input of features matrix and label matrix, as well as the desired
 * weights and bias values for single-layer neural network model. Parallel
 * model training enabled.
 */
public class ParallelModel implements Runnable {

  private Matrix X;
  private Matrix Y;
  private int numRows;
  private int numColumns;
  private int numLabels;  
  private Matrix W;
  private Matrix B;

  public ParallelModel(Matrix X, Matrix Y) {
    this.X = X;
    this.Y = Y;
    this.numRows = X.entries.length;
    this.numColumns = X.entries[0].length;

    Set<Double> labelSet = new HashSet<Double>();
    for (int i=0; i<this.numRows; i++) {
      labelSet.add(Y.entries[i][0]);
    }
    this.numLabels = labelSet.size();

    this.W = Matrix.random(this.numLabels, this.numColumns, 0.001);
    this.B = Matrix.random(this.numLabels, 1, 0.001);
  }

  public void run() {
    double loss = 0;

    for (int i=0; i<this.numRows; i++) {
      int label = (int) this.Y.entries[i][0];

      Matrix rowData = new Matrix(this.numColumns, 1);
      for (int j=0; j<this.numColumns; j++) {
        rowData.entries[j][0] = this.X.entries[i][j];
      }

      Matrix result = this.W.times(rowData).plus(this.B);
      Matrix normResult = result.softmax();

      int labelIndex = normResult.argmax()[0];
      loss += -Math.log(normResult.entries[labelIndex][0]);
      System.out.println(java.lang.Thread.currentThread().getName() + " "
          + loss / (double) i);

      Matrix labelOneHot = new Matrix(this.numLabels, 1);
      labelOneHot.entries[label][0] = 1.0;
      result = result.minus(labelOneHot);

      Matrix columnData = rowData.transpose();
      Matrix gradient = result.times(columnData);

      Matrix learningRate = Matrix.diagonal(this.numLabels, 0.001);
      gradient = learningRate.times(gradient);
      this.W = this.W.minus(gradient);
    }
  }

  public static void main(String[] args) throws Exception {
    Matrix[] rawData = PreProcessing.readCSV("data.csv");
    Matrix X = PreProcessing.scaleFeatures(rawData[0]);
    Matrix Y = rawData[1];

    Matrix[] cleanData = PreProcessing.evenOddSplit(X, Y);
    Matrix X1 = cleanData[0];
    Matrix Y1 = cleanData[1];
    Matrix X2 = cleanData[2];
    Matrix Y2 = cleanData[3];

    ParallelModel p1 = new ParallelModel(X1, Y1);
    java.lang.Thread t1 = new java.lang.Thread(p1);

    ParallelModel p2 = new ParallelModel(X2, Y2);
    java.lang.Thread t2 = new java.lang.Thread(p1);

    t1.start();
    t2.start();

    t1.join();
    t2.join();

    Matrix W = p1.W.plus(p2.W);
    Matrix B = p1.B.plus(p2.B);
    Matrix divisor = Matrix.diagonal(W.entries.length, 0.5);
    W = divisor.times(W);
    B = divisor.times(B);
    
    p1.W.show();
  }
}









