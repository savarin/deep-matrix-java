public class Matrix {
  private int m;
  private int n;
  private float[][] entries;

  public Matrix(float[][] entries) {
    m = entries.length;
    n = entries[0].length;
    this.entries = new float[m][n];

    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
        this.entries[i][j] = entries[i][j];
      }
    }
  }
  
  public Matrix dot(Matrix B) {
    Matrix A = this;
    if (A.n != B.m) throw new RuntimeException("Improper matrix shapes.");

    float [][] c = new float[A.m][B.n];
    Matrix C = new Matrix(c);

    for (int i=0; i<A.m; i++) {
      for (int j=0; j<B.n; j++) {
        for (int k=0; k<A.n; k++) {
          C.entries[i][j] += A.entries[i][k] * B.entries[k][j];
        }
      }
    }

    return C;
  }

  public void print() {
    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
        System.out.printf("%.1f ", entries[i][j]);
      }
      System.out.println();
    }
  }

  public static void main(String[] args) {
    float [][] a = {{1, 2}, {2, 1}};
    float [][] b = {{1, 0, 3}, {0, 1, 3}, {0, 1, 3}};

    Matrix A = new Matrix(a);
    Matrix B = new Matrix(b);
    Matrix C = new Matrix(a);

    C = A.dot(B);

    C.print();
  }
}