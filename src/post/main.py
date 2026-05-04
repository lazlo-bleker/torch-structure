from obj_functions.fairness import graph_post_process as graph_post_process_1
from obj_functions.self_supporting import graph_post_process as graph_post_process_2

from post.core import post_process, PostProcessConfig
from post.custom import (
    export_aggragate_img,
    export_aggragate_vtp,
    export_assembly_states_img,
    export_assembly_states_vtp,
)

def graph_post_process(graph):
    graph = graph_post_process_1(graph)
    graph = graph_post_process_2(graph)
    return graph

def post_main(dir_results):
    # first_iter_name = "state_0000.json"
    # last_iter_name = "state_0200.json"

    # A. Export aggregate state of all iterations
    config = PostProcessConfig(
        dir=dir_results,
        label="opt",
        # export_img=export_aggragate_img,
        export_vtp=export_aggragate_vtp,
        post_process_func=graph_post_process,
    )
    post_process(config)

    # # B. Export assembly states of first iteration
    # config = PostProcessConfig(
    #     dir=dir_results,
    #     label="initial",
    #     files=[first_iter_name],
    #     export_img=export_assembly_states_img,
    #     export_vtp=export_assembly_states_vtp,
    #     post_process_func=graph_post_process,
    # )
    # post_process(config)

    # # C. Export assembly states of last iteration
    # config = PostProcessConfig(
    #     dir=dir_results,
    #     label="final",
    #     files=[last_iter_name],
    #     export_img=export_assembly_states_img,
    #     export_vtp=export_assembly_states_vtp,
    #     post_process_func=graph_post_process,
    # )
    # post_process(config)


if __name__ == "__main__":
    post_main()
