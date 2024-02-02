import numpy as np
import numpy as np

def dot_product(matrix1, matrix2):
      out=np.zeros_like(matrix1)
      for k in range(matrix1.shape[2]):
        out[:,:,k]=np.dot(matrix1[:,:,k],matrix2[:,:,k]) 
      return out           
# Create two matrices of shape (12, 12, 4)
matrix1 = np.random.rand(12, 12, 4)
matrix2 = np.random.rand(12, 12, 4)

# Compute the dot product of the two matrices
result = dot_product(matrix1, matrix2)

# Print the result
print(result.shape)
