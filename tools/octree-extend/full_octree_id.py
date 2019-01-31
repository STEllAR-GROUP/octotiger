import argparse


NDIM = 3


def to_id(indices, levels):
    id = 1
    for l in range(0, levels):
        for d in range(0, NDIM):
            id <<= 1
            id |= (indices[d] >> l) & 1
    return id


def parse_id(id):
    indices = [0] * NDIM
    level = 0
    
    while id != 1:
        print('{} {:o}'.format(level, id))
        for d in range(NDIM - 1, -1, -1):
            indices[d] <<= 1
            indices[d] |= (id & 1)
            id >>= 1
        level += 1
        
    return indices, level


def main():
    parser = argparse.ArgumentParser(
        description='Generate Octree IDs for elements of a subgrid')
    parser.add_argument('base', type=str, help='sub-grid id')
    parser.add_argument('additional_levels', type=int, help='number of additional levels')
    args = parser.parse_args()

    subgrid_id = int(args.base, 8)
    additional_levels = args.additional_levels

    subgrid_indices, base_level = parse_id(subgrid_id)

    print("base_level is", base_level)

    for i in range(0, 1 << additional_levels):
        for j in range(0, 1 << additional_levels):
            for k in range(0, 1 << additional_levels):
                indices = (
                    (subgrid_indices[0] << additional_levels) + i,
                    (subgrid_indices[1] << additional_levels) + j,
                    (subgrid_indices[2] << additional_levels) + k
                )
                new_id = to_id(indices, base_level + additional_levels)
                print('the ({},{},{}) cell in sub-grid {} has full octree id {:o}'.format(
                    i, j, k, args.base, new_id))


if __name__ == '__main__':
    main()