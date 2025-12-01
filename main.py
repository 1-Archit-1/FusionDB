from evaluator import Evaluator

results_dir = "results/"

eval_duck = Evaluator(meta_method='duck')
eval_duck.evaluate()
eval_duck.export_results(output_dir=f"{results_dir}duck")
eval_duck.export_averaged_table(output_file=f"{results_dir}duck/averaged_results.csv")
eval_duck.plot_results()

eval_idx = Evaluator(meta_method='index')
eval_idx.evaluate()
eval_idx.export_results(output_dir=f"{results_dir}index")
eval_idx.export_averaged_table(output_file=f"{results_dir}index/averaged_results.csv")
eval_idx.plot_results()
