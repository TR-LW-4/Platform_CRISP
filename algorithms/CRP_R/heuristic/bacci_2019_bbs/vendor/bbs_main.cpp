/*
 * Minimal driver for the BBS (Bounded Beam Search) heuristic.
 *
 * Bacci, Mattia, Ventura (2020) BC-RBRP — heuristic-only build.
 * No Gurobi dependency.
 *
 * Usage:  bbs_heuristic <instance_file> [time_limit_sec]
 *   Default time limit: 5.0 seconds.
 *
 * Input format:
 *   w h n
 *   k item1 item2 ...    (one line per stack; items bottom-to-top, 1-indexed)
 *   ...                  (repeat w times)
 *
 * Output (stdout):
 *   reshuffles=<N>
 */

#include <stdio.h>
#include <stdlib.h>

/* Modified rBRP_BSheu accepts a timelimit parameter instead of using the
   hardcoded 1.0 s default from the original source. */
int rBRP_BSheu(int n, int w, int h, int **yard, int ***bbs_solution,
               double timelimit);

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: bbs_heuristic <instance_file> [time_limit_sec]\n");
        return 1;
    }

    double timelimit = 5.0;
    if (argc >= 3)
        sscanf(argv[2], "%lf", &timelimit);

    /* ── Read instance ───────────────────────────────────────────────── */
    FILE *iF = fopen(argv[1], "r");
    if (!iF) {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        return 1;
    }

    int n, w, h;
    fscanf(iF, "%d %d %d", &w, &h, &n);

    int **A = (int **)malloc(w * sizeof(int *));
    for (int i = 0; i < w; ++i) {
        A[i] = (int *)malloc(h * sizeof(int));
        int k;
        fscanf(iF, "%d", &k);
        for (int j = 0; j < k; ++j)
            fscanf(iF, "%d", &A[i][j]);
        for (int j = k; j < h; ++j)
            A[i][j] = -1;
    }
    fclose(iF);

    /* ── Allocate solution array (required by rBRP_BSheu) ───────────── */
    int ***solution = (int ***)malloc((n + 1) * sizeof(int **));
    for (int i = 0; i < n + 1; i++) {
        solution[i] = (int **)malloc(w * sizeof(int *));
        for (int j = 0; j < w; j++)
            solution[i][j] = (int *)malloc(h * sizeof(int));
    }

    /* ── Run BBS heuristic ───────────────────────────────────────────── */
    int ub = rBRP_BSheu(n, w, h, A, solution, timelimit);
    printf("reshuffles=%d\n", ub);

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    for (int i = 0; i < w; i++) free(A[i]);
    free(A);
    for (int i = 0; i < n + 1; i++) {
        for (int j = 0; j < w; j++) free(solution[i][j]);
        free(solution[i]);
    }
    free(solution);

    return 0;
}
