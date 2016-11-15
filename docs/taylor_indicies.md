
 The indicies defined in the arrays `taylor_consts::map2`, `taylor_consts::map3`,
 and `taylor_consts::map4` are set up in a way, such that using them in loops
 which are structured as

    taylor<5, T> x;
    x() = ...
    for (int a = 0; a != NDIM; ++a) {
        x(a) = ...;
        for (int b = a; b != NDIM; ++b) {
            x(a, b) = ...;
            for (int c = b; c != NDIM; ++c) {
                x(a, b, c) = ...;
                for (int d = c; d != NDIM; ++d) {
                    x(a, b, c, d) = ...;
            }
        }
    }

 the expressions `x(a)`, `x(a, b)`, `x(a, b, c)`, and `x(a, b, c, d)` evaluate to
 monotonically increasing indicies into the one dimensional array representing
 the storage inside `x.data`:

 Zero-dimensional index:

  | x() | x.data[0] |
  |-----|-----------|

 One-dimensional indicies:

  | a |   x(a)    |
  |---|-----------|
  | 0 | x.data[1] |
  | 1 | x.data[2] |
  | 2 | x.data[3] |

Two-dimensional indicies:
 
  | | a b |  x(a, b)  | | a b |  x(a, b)  | | a b |  x(a, b)  |
  |-|-----|-----------|-|-----|-----------|-|-----|-----------|
  | | 0 0 | x.data[4] |*| 1 0 |           |*| 2 0 |           |
  | | 0 1 | x.data[5] | | 1 1 | x.data[7] |*| 2 1 |           |
  | | 0 2 | x.data[6] | | 1 2 | x.data[8] | | 2 2 | x.data[9] |
 
Three-dimensional indicies:
 
  | | a b c | x(a, b, c) | | a b c | x(a, b, c) | | a b c | x(a, b, c) |
  |-|-------|------------|-|-------|------------|-|-------|------------|
  | | 0 0 0 | x.data[10] |*| 1 0 0 |            |*| 2 0 0 |            |
  | | 0 0 1 | x.data[11] |*| 1 0 1 |            |*| 2 0 1 |            |
  | | 0 0 2 | x.data[12] |*| 1 0 2 |            |*| 2 0 2 |            |
  |*| 0 1 0 |            |*| 1 1 0 |            |*| 2 1 0 |            |
  | | 0 1 1 | x.data[13] | | 1 1 1 | x.data[16] |*| 2 1 1 |            |
  | | 0 1 2 | x.data[14] | | 1 1 2 | x.data[17] |*| 2 1 2 |            |
  |*| 0 2 0 |            |*| 1 2 0 |            |*| 2 2 0 |            |
  |*| 0 2 1 |            |*| 1 2 1 |            |*| 2 2 1 |            |
  | | 0 2 2 | x.data[15] | | 1 2 2 | x.data[18] | | 2 2 2 | x.data[19] |
  
 Four-dimensional indicies:
 
  | | a b c d | x(a, b, c, d) | | a b c d | x(a, b, c, d) | | a b c d | x(a, b, c, d) |
  |-|---------|---------------|-| --------|---------------|-|---------|---------------|
  | | 0 0 0 0 |  x.data[20]   |*| 1 0 0 0 |               |*| 2 0 0 0 |               |
  | | 0 0 0 1 |  x.data[21]   |*| 1 0 0 1 |               |*| 2 0 0 1 |               |
  | | 0 0 0 2 |  x.data[22]   |*| 1 0 0 2 |               |*| 2 0 0 2 |               |
  |*| 0 0 1 0 |               |*| 1 0 1 0 |               |*| 2 0 1 0 |               |
  | | 0 0 1 1 |  x.data[23]   |*| 1 0 1 1 |               |*| 2 0 1 1 |               |
  | | 0 0 1 2 |  x.data[24]   |*| 1 0 1 2 |               |*| 2 0 1 2 |               |
  |*| 0 0 2 0 |               |*| 1 0 2 0 |               |*| 2 0 2 0 |               |
  |*| 0 0 2 1 |               |*| 1 0 2 1 |               |*| 2 0 2 1 |               |
  | | 0 0 2 2 |  x.data[25]   |*| 1 0 2 2 |               |*| 2 0 2 2 |               |
  |*| 0 1 0 0 |               |*| 1 1 0 0 |               |*| 2 1 0 0 |               |
  |*| 0 1 0 1 |               |*| 1 1 0 1 |               |*| 2 1 0 1 |               |
  |*| 0 1 0 2 |               |*| 1 1 0 2 |               |*| 2 1 0 2 |               |
  |*| 0 1 1 0 |               |*| 1 1 1 0 |               |*| 2 1 1 0 |               |
  | | 0 1 1 1 |  x.data[26]   | | 1 1 1 1 |   x.data[30]  |*| 2 1 1 1 |               |
  | | 0 1 1 2 |  x.data[27]   | | 1 1 1 2 |   x.data[31]  |*| 2 1 1 2 |               |
  |*| 0 1 2 0 |               |*| 1 1 2 0 |               |*| 2 1 2 0 |               |
  |*| 0 1 2 1 |               |*| 1 1 2 1 |               |*| 2 1 2 1 |               |
  | | 0 1 2 2 |  x.data[28]   | | 1 1 2 2 |   x.data[32]  |*| 2 1 2 2 |               |
  |*| 0 2 0 0 |               |*| 1 2 0 0 |               |*| 2 2 0 0 |               |
  |*| 0 2 0 1 |               |*| 1 2 0 1 |               |*| 2 2 0 1 |               |
  |*| 0 2 0 2 |               |*| 1 2 0 2 |               |*| 2 2 0 2 |               |
  |*| 0 2 1 0 |               |*| 1 2 1 0 |               |*| 2 2 1 0 |               |
  |*| 0 2 1 1 |               |*| 1 2 1 1 |               |*| 2 2 1 1 |               |
  |*| 0 2 1 2 |               |*| 1 2 1 2 |               |*| 2 2 1 2 |               |
  |*| 0 2 2 0 |               |*| 1 2 2 0 |               |*| 2 2 2 0 |               |
  |*| 0 2 2 1 |               |*| 1 2 2 1 |               |*| 2 2 2 1 |               |
  | | 0 2 2 2 | x.data[29]    | | 1 2 2 2 |   x.data[33]  | | 2 2 2 2 |   x.data[34]  |

