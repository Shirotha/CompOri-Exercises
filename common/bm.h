enum BM_State
{
    READY=0,
    RUNNING=1,
    FINISHED=2
};

struct BM_Data
{
    enum BM_State state;
    double start;
    double end;
};

int bm_start(struct BM_Data* data);
int bm_end(struct BM_Data* data);

int bm_batch(struct BM_Data* data, int size, int (*func)(int));

int bm_print(struct BM_Data* data);
int bm_print_batch(struct BM_Data* data, int size);