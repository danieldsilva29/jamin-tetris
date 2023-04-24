from statistics import mean
def get_stats(game_field, game_figure, game_height, game_width):
    gaps = 0
    heights = []

    seen_tiles = False
    for i in range(game_height):
        zeroes = 0
        for j in range(game_width):
            if game_field[i][j] == 0:
                zeroes += 1
            else:
                seen_tiles = True
                heights.append((i, j))
        if seen_tiles:
            gaps += zeroes
    
    
    # Calculate mean heights

    mean_height = mean([x[0] for x in heights])
    # Calculate stdev of heights
    # Calculate range of heights
    # Calculate max consecutive height diff