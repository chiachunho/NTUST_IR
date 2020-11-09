import sys


def main():
    n = int(sys.stdin.readline())
    total_map_score = 0
    for i in range(n):
        query_result = sys.stdin.readline().split()
        query_relative = sys.stdin.readline().split()

        single_map_score = 0
        relative_amount = 0

        for index in range(len(query_result)):
            if query_result[index] in query_relative:
                relative_amount += 1
                single_map_score += relative_amount / (index + 1)

                if relative_amount == len(query_relative):
                    break

        total_map_score += single_map_score / relative_amount

    average_map_score = total_map_score / n
    print(round(average_map_score, 4))


if __name__ == '__main__':
    main()
