import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class GenericModel {

  public static List<List<String>> readCSV(String fileName)
      throws FileNotFoundException, IOException {
    BufferedReader inputData = new BufferedReader(new FileReader(new File(fileName)));
    List<List<String>> data = new ArrayList<List<String>>();
    String line = "";

    while ((line = inputData.readLine()) != null) {
      data.add(Arrays.asList(line.split(",")));
    }

    return data;
  }

  public static void allRows()
      throws FileNotFoundException, IOException {
    List<List<String>> data = readCSV("data.csv");

    int numRows = data.size();
    int numColumns = data.get(0).size();

    Set<String> labelSet = new HashSet<String>();

    for (int i=0; i<numRows; i++) {
      labelSet.add(data.get(i).get(0));
    }

    int numLabels = labelSet.size();

    double[][] w = {{0.00262025, 0.00158684, 0.00278127, 0.00459317},
        {0.00321001, 0.00518393, 0.00261943, 0.00976085}};
    Matrix W = new Matrix(w);

    double[][] b = {{0.00732815}, {0.00115274}};
    Matrix B = new Matrix(b);

    double loss = 0;

    for (int i=1; i<numRows; i++) {
      int label = Integer.parseInt(data.get(i).get(0));

      double[][] features = new double[numColumns-1][1];
      for (int j=1; j<numColumns; j++) {
        features[j-1][0] = Double.parseDouble(data.get(i).get(j));
      }

      Matrix X = new Matrix(features);

      Matrix result = new Matrix(numLabels, 1);
      result = W.times(X).plus(B);
      result = result.softmax();

      int labelIndex = result.argmax()[0];
      loss += -Math.log(result.entries[labelIndex][0]);
    }

    System.out.println(loss / numRows);
  }

  public static void naiveOptimization()
      throws FileNotFoundException, IOException {
    List<List<String>> data = readCSV("data.csv");

    int numRows = data.size();
    int numColumns = data.get(0).size();

    Set<String> labelSet = new HashSet<String>();

    for (int i=0; i<numRows; i++) {
      labelSet.add(data.get(i).get(0));
    }

    int numLabels = labelSet.size();

    double minLoss = 1000.0;
    Matrix bestWeights = new Matrix(numLabels, numColumns-1);
    Matrix bestBias = new Matrix(numLabels, 1);

    for (int i=0; i<1000; i++) {
      Matrix W = new Matrix(numLabels, numColumns-1);
      W = Matrix.random(numLabels, numColumns-1);

      Matrix B = new Matrix(numLabels, 1);
      B = Matrix.random(numLabels, 1);

      double loss = 0;

      for (int j=1; j<numRows; j++) {
        int label = Integer.parseInt(data.get(j).get(0));

        double[][] features = new double[numColumns-1][1];
        for (int k=1; k<numColumns; k++) {
          features[k-1][0] = Double.parseDouble(data.get(j).get(k));
        }

        Matrix X = new Matrix(features);

        Matrix result = new Matrix(numLabels, 1);
        result = W.times(X).plus(B);
        result = result.softmax();

        int labelIndex = result.argmax()[0];
        loss += -Math.log(result.entries[labelIndex][0]);
      }

      loss = loss / numRows;

      if (loss < minLoss) {
        minLoss = loss;
        bestWeights = W;
        bestBias = B;
        System.out.printf("loss %.9f loop %d", loss, i);
        System.out.println();
      }
    }
  }

  public static void gradientOptimization()
      throws FileNotFoundException, IOException {
    List<List<String>> data = readCSV("data.csv");

    int numRows = data.size();
    int numColumns = data.get(0).size();

    Set<String> labelSet = new HashSet<String>();

    for (int i=0; i<numRows; i++) {
      labelSet.add(data.get(i).get(0));
    }

    int numLabels = labelSet.size();

    Matrix W = new Matrix(numLabels, numColumns-1);
    W = Matrix.random(numLabels, numColumns-1);

    Matrix B = new Matrix(numLabels, 1);
    B = Matrix.random(numLabels, 1);

    double loss = 0;

    for (int i=1; i<numRows; i++) {
      int label = Integer.parseInt(data.get(i).get(0));

      double[][] features = new double[numColumns-1][1];
      for (int j=1; j<numColumns; j++) {
        features[j-1][0] = Double.parseDouble(data.get(i).get(j));
      }

      Matrix X = new Matrix(features);

      Matrix result = new Matrix(numLabels, 1);
      result = W.times(X).plus(B);
      result = result.softmax();

      int labelIndex = result.argmax()[0];
      loss += -Math.log(result.entries[labelIndex][0]);
      System.out.println(loss / (double) i);

      Matrix labelOneHot = new Matrix(numLabels, 1);
      labelOneHot.entries[label][0] = 1.0;
      result = W.times(X).plus(B);
      result = result.minus(labelOneHot);

      Matrix T = new Matrix(1, numColumns-1);
      T = X.transpose();
      Matrix gradient = new Matrix(numLabels, numColumns-1);
      gradient = result.times(T);

      Matrix learningRate = new Matrix(numLabels, numLabels);
      learningRate = Matrix.identity(numLabels, 0.001);
      gradient = learningRate.times(gradient);
      W = W.minus(gradient);
    }
  }

  public static void main(String[] args)
    throws FileNotFoundException, IOException {
    // allRows();
    // naiveOptimization();
    gradientOptimization();      
  }
}


