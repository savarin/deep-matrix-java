import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class Model {
  
  public static void main(String[] args) throws FileNotFoundException, IOException {

    BufferedReader inputData = new BufferedReader(new FileReader(new File("data.csv")));
    List<List<String>> data = new ArrayList<List<String>>();
    String line = "";

    while ((line = inputData.readLine()) != null) {
      data.add(Arrays.asList(line.split(",")));
    }

    int numRows = data.size();
    int numColumns = data.get(0).size();
    
    double minLoss = 1000.0;
    Matrix bestWeights = new Matrix(2, 4);
    Matrix bestBias = new Matrix(2, 1);

    for (int i=0; i<1000; i++) {

      double loss = 0;
  
      Matrix W = new Matrix(2, 4);
      Matrix b = new Matrix(2, 1);

      for (int j=0; j<numRows; j++) {
        int label = Integer.parseInt(data.get(j).get(0));
    
        double[][] features = new double[numColumns-1][1];
        for (int k=1; k<numColumns; k++) {
          features[k-1][0] = Double.parseDouble(data.get(0).get(k));
        }
    
        Matrix X = new Matrix(features);
        
        W = Matrix.random(2, 4);
        b = Matrix.random(2, 1);
    
        Matrix result = new Matrix(2, 1);
        result = W.times(X).plus(b);
        
        result = result.softmax();
        
        int labelIndex = result.softmax().argmax()[0];
    
        loss += -Math.log(result.entries[labelIndex][0]);
      
        if (loss < minLoss) {
          minLoss = loss;
          bestWeights = W;
          bestBias = b;
          System.out.printf("loss %.3f loop %d", loss, i);
          System.out.println();
        }
      }
    }
  }
}


