def main():
    import numpy as np
    a = np.array([[1,2,3],
                  [2,3,4],
                  [4,5,6],
                  [7,8,9]])
    b = a.reshape(a.shape[0], 1, a.shape[1])
    print(a.shape)
    print(b.shape)
    print((a-b).shape)
    print(np.sqrt(np.einsum('ijk, ijk->ij', a-b, a-b)))

if __name__ == '__main__':
    main()
