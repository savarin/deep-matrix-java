
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class Matrix {
  private int m;
  private int n;
  public double[][] entries;

  /** First constructor for Matrix object. Creates an empty matrix with m rows and n columns. */
  public Matrix(int m, int n) {
    this.m = m;
    this.n = n;
    this.entries = new double[m][n];
  }

  /** Second constructor for Matrix object. Creates matrix with values as per entries array. */
  public Matrix(double[][] entries) {
    this.m = entries.length;
    this.n = entries[0].length;
    this.entries = new double[this.m][this.n];

    for (int i = 0; i < this.m; i++) {
      for (int j = 0; j < this.n; j++) {
        this.entries[i][j] = entries[i][j];
      }
    }
  }

  /** Selects row of specified index. */
  public Matrix row(int rowIndex) {
    Matrix rowData = new Matrix(1, this.n);

    for (int j = 0; j < this.n; j++) {
      rowData.entries[0][j] = this.entries[rowIndex][j];
    }

    return rowData;
  }

  /** Selects column of specified index. */
  public Matrix column(int columnIndex) {
    Matrix columnData = new Matrix(this.m, 1);

    for (int i = 0; i < this.m; i++) {
      columnData.entries[i][0] = this.entries[i][columnIndex];
    }

    return columnData;
  }

  /**
   * Random mxn matrix with Gaussian distributed values ~N(0, scalar).
   *
   * @param m Row size of matrix.
   * @param n Column size of matrix.
   * @param scalar Standard deviation multiple.
   * @return Matrix Random mxn matrix.
   */
  public static Matrix random(int m, int n, double scalar) {
    Matrix R = new Matrix(m, n);
    Random randomClass = new Random();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        R.entries[i][j] = randomClass.nextGaussian() * scalar;
      }
    }

    return R;
  }

  /**
   * Diagonal nxn matrix with value of scalar on diagonal entries. Identity matrix if scalar = 1.
   *
   * @param n Size of matrix.
   * @param scalar Value on diagonal elements of the matrix.
   * @return Matrix Diagonal matrix of size n with scalar as diagonal entries.
   */
  public static Matrix diagonal(int n, double scalar) {
    Matrix I = new Matrix(n, n);

    for (int i = 0; i < n; i++) {
      I.entries[i][i] = scalar;
    }

    return I;
  }

  /** Addition operation for two matrices. */
  public Matrix plus(Matrix B) {
    Matrix A = this;
    Matrix C = new Matrix(A.m, A.n);

    if (A.n != B.n || A.m != B.m) throw new RuntimeException("Improper matrix shape.");

    for (int i = 0; i < A.m; i++) {
      for (int j = 0; j < A.n; j++) {
        C.entries[i][j] = A.entries[i][j] + B.entries[i][j];
      }
    }

    return C;
  }

  /** Subtraction operation for two matrices. */
  public Matrix minus(Matrix B) {
    Matrix A = this;
    Matrix C = new Matrix(A.m, A.n);

    if (A.n != B.n || A.m != B.m) throw new RuntimeException("Improper matrix shape.");

    for (int i = 0; i < A.m; i++) {
      for (int j = 0; j < A.n; j++) {
        C.entries[i][j] = A.entries[i][j] - B.entries[i][j];
      }
    }

    return C;
  }

  /** Matrix multiplication operation for two matrices. */
  public Matrix times(Matrix B) {
    Matrix A = this;
    Matrix C = new Matrix(A.m, B.n);

    if (A.n != B.m) throw new RuntimeException("Improper matrix shape.");

    for (int i = 0; i < A.m; i++) {
      for (int j = 0; j < B.n; j++) {
        for (int k = 0; k < A.n; k++) {
          C.entries[i][j] += A.entries[i][k] * B.entries[k][j];
        }
      }
    }

    return C;
  }

  /** Equality operation for two matrices. */
  public boolean equals(Matrix B) {
    Matrix A = this;

    if (A.n != B.n) return false;
    if (A.m != B.m) return false;

    for (int i = 0; i < A.m; i++) {
      for (int j = 0; j < A.n; j++) {
        if (A.entries[i][j] != B.entries[i][j]) {
          return false;
        }
      }
    }

    return true;
  }

  /** Matrix transpose operation. */
  public Matrix transpose() {
    Matrix C = new Matrix(this.n, this.m);

    for (int i = 0; i < this.m; i++) {
      for (int j = 0; j < this.n; j++) {
        C.entries[j][i] = this.entries[i][j];
      }
    }

    return C;
  }

  /** Softmax mapping to all entries of the matrix. */
  public Matrix softmax() {
    double expEntry;
    double expTotal = 0;
    Matrix C = new Matrix(this.m, this.n);

    for (int i = 0; i < this.m; i++) {
      for (int j = 0; j < this.n; j++) {
        expEntry = Math.exp(this.entries[i][j]);
        expTotal += expEntry;
        C.entries[i][j] = expEntry;
      }
    }

    for (int i = 0; i < this.m; i++) {
      for (int j = 0; j < this.n; j++) {
        C.entries[i][j] = C.entries[i][j] / expTotal;
      }
    }

    return C;
  }

  /** Retains only the positive entries of a matrix. */
  public Matrix relu() {
    Random randomClass = new Random();
    Matrix C = new Matrix(this.m, this.n);

    for (int i = 0; i < this.m; i++) {
      for (int j = 0; j < this.n; j++) {
        if (this.entries[i][j] > 0) {
          C.entries[i][j] = this.entries[i][j];
        }
      }
    }

    return C;
  }

  /** Randomly sets entries to zero by pre-specified probability. */
  public Matrix dropout(double probability) {
    Random randomClass = new Random();
    Matrix C = new Matrix(this.m, this.n);

    for (int i = 0; i < this.m; i++) {
      for (int j = 0; j < this.n; j++) {
        if (randomClass.nextFloat() > probability) {
          C.entries[i][j] = this.entries[i][j];
        }
      }
    }

    return C;
  }

  /** Returns indices of the maximum value of the matrix. */
  public int[] argmax() {
    double maxEntry = this.entries[0][0];
    int[] maxIndex = {0, 0};

    for (int i = 0; i < this.m; i++) {
      for (int j = 0; j < this.n; j++) {
        if (this.entries[i][j] > maxEntry) {
          maxIndex[0] = i;
          maxIndex[1] = j;
          maxEntry = this.entries[i][j];
        }
      }
    }

    return maxIndex;
  }

  /** Returns count of unique entries. */
  public int unique() {
    Set<Double> labelSet = new HashSet<Double>();
    for (int i = 0; i < this.m; i++) {
      for (int j = 0; j < this.n; j++) {
        labelSet.add(this.entries[i][0]);
      }
    }

    return labelSet.size();
  }

  /** Returns matrix shape. */
  public int[] shape() {
    int[] size = new int[2];
    size[0] = this.m;
    size[1] = this.n;

    return size;
  }

  /** Prints matrix entries. */
  public void show() {
    for (int i = 0; i < this.m; i++) {
      for (int j = 0; j < this.n; j++) {
        System.out.printf("%.6f ", this.entries[i][j]);
      }
      System.out.println();
    }
  }
}
