
public class Test {

  public static boolean testMatrixConstructor() {
    Matrix A = new Matrix(2, 2);
    Matrix B = new Matrix(new double[2][2]);
    return A.equals(B);
  }

  public static boolean testMatrixDiagonal() {
    Matrix A = Matrix.diagonal(2, 1.);
    Matrix B = new Matrix(new double[2][2]);
    B.entries[0][0] = 1;
    B.entries[1][1] = 1;
    return A.equals(B);
  }

  public static boolean testMatrixPlus() {
    Matrix A = new Matrix(2, 2);
    Matrix B = new Matrix(2, 2);
    Matrix C = new Matrix(2, 2);

    for (int i = 0; i < A.shape()[0]; i++) {
      for (int j = 0; j < A.shape()[1]; j++) {
        A.entries[i][j] = 1.;
        B.entries[i][j] = 2.;
        C.entries[i][j] = 3.;
      }  
    }
    return (A.plus(B)).equals(C);
  }

  public static boolean testMatrixMinus() {
    Matrix A = new Matrix(2, 2);
    Matrix B = new Matrix(2, 2);
    Matrix C = new Matrix(2, 2);

    for (int i = 0; i < A.shape()[0]; i++) {
      for (int j = 0; j < A.shape()[1]; j++) {
        A.entries[i][j] = 3.;
        B.entries[i][j] = 2.;
        C.entries[i][j] = 1.;
      }  
    }
    return (A.minus(B)).equals(C);
  }

  public static boolean testMatrixTimes() {
    Matrix A = Matrix.diagonal(2, 2.);
    Matrix B = Matrix.diagonal(2, 2.);
    Matrix C = Matrix.diagonal(2, 4.);
    return (A.times(B)).equals(C);
  }

  public static boolean testMatrixTranspose() {
    Matrix A = new Matrix(new double[2][2]);
    A.entries[0][1] = 1;
    Matrix B = new Matrix(new double[2][2]);
    B.entries[1][0] = 1;
    return A.transpose().equals(B);
  }

  public static boolean testMatrixSoftmax() {
    Matrix A = new Matrix(new double[2][2]);
    Matrix B = new Matrix(new double[2][2]);
    for (int i = 0; i < A.shape()[0]; i++) {
      for (int j = 0; j < A.shape()[1]; j++) {
        A.entries[i][j] = 1.;
        B.entries[i][j] = 0.25;
      }  
    }
    return (A.softmax()).equals(B);
  }

  public static boolean testMatrixRelu() {
    Matrix A = new Matrix(new double[2][2]);
    A.entries[0][0] = 1.;
    A.entries[0][1] = -1.;
    A.entries[1][0] = -1.;
    A.entries[1][1] = 1.;
    Matrix B = Matrix.diagonal(2, 1.);
    return (A.relu()).equals(B);
  }

  public static void main(String[] args) {
    assert testMatrixConstructor();
    assert testMatrixDiagonal();
    assert testMatrixPlus();
    assert testMatrixMinus();
    assert testMatrixTimes();
    assert testMatrixTranspose();
    assert testMatrixSoftmax();
    assert testMatrixRelu();
  }
}