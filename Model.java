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

  System.out.println(data);
  }
}


