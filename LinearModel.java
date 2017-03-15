
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Represents input of features matrix and label matrix, as well as the desired
 * weights and bias values for single-layer neural network model.
 */
public class LinearModel {

  private Matrix X;
  private Matrix Y;  
  private int numRows;
  private int numColumns;
  private int numLabels;
  private Matrix W;
  private Matrix B;

  /**
   * Constructor for LinearModel class, with input feature matrix X and label
   * matrix Y.
   */
  public LinearModel(Matrix X, Matrix Y) {
    this.X = X;
    this.Y = Y;
    this.numRows = X.entries.length;
    this.numColumns = X.entries[0].length;

    Set<Double> labelSet = new HashSet<Double>();
    for (int i=0; i<this.numRows; i++) {
      labelSet.add(Y.entries[i][0]);
    }
    this.numLabels = labelSet.size();

    this.W = new Matrix(this.numLabels, this.numColumns);
    this.B = new Matrix(this.numLabels, 1);
  }

  /**
   * Naive model training by selecting the weights-bias matrix pair that 
   * produces the smallest lost.
   */
  public void naiveOptimization() {
    double minLoss = 1000.0;

    for (int k=0; k<1000; k++) {
      Matrix randomW = Matrix.random(this.numLabels, this.numColumns);
      Matrix randomB = Matrix.random(this.numLabels, 1);

      double loss = 0;

      for (int i=0; i<this.numRows; i++) {
        int label = (int) this.Y.entries[i][0];

        double[][] features = new double[this.numColumns][1];
        for (int j=0; j<this.numColumns; j++) {
          features[j][0] = this.X.entries[i][j];
        }

        Matrix rowData = new Matrix(features);

        Matrix result = randomW.times(rowData).plus(randomB);
        Matrix normResult = result.softmax();

        int labelIndex = normResult.argmax()[0];
        loss += -Math.log(normResult.entries[labelIndex][0]);
      }

      loss = loss / numRows;

      if (loss < minLoss) {
        minLoss = loss;
        this.W = randomW;
        this.B = randomB;
        System.out.printf("loss %.9f loop %d", loss, i);
        System.out.println();
      }
    }
  }

  /**
   * Model training by gradient descent.
   */
  public void gradientOptimization() {
    double loss = 0;

    this.W = Matrix.random(this.numLabels, this.numColumns);
    this.B = Matrix.random(this.numLabels, 1);

    for (int i=0; i<this.numRows; i++) {
      int label = (int) this.Y.entries[i][0];

      double[][] features = new double[this.numColumns][1];
      for (int j=0; j<this.numColumns; j++) {
        features[j][0] = this.X.entries[i][j];
      }

      Matrix rowData = new Matrix(features);

      Matrix result = this.W.times(rowData).plus(this.B);
      Matrix normResult = result.softmax();

      int labelIndex = normResult.argmax()[0];
      loss += -Math.log(normResult.entries[labelIndex][0]);
      System.out.println(loss / (double) i);

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
    Matrix X = PreProcessing.scaleMatrix(rawData[0]);
    Matrix Y = rawData[1];

    LinearModel l1 = new LinearModel(X, Y);
    // l1.naiveOptimization();
    l1.gradientOptimization();
  }
}