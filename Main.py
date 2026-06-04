from PipelineManager import PipelineManager

config = {
    "data_source": "DisasterTweets.csv",
    "output_dir": "runtime_results/manual_run"
}
pipeline = PipelineManager(config)

results = pipeline.run()

# Retrieve the fact-check results
if hasattr(pipeline, 'fact_checker'):
    fact_check_results = pipeline.fact_checker.load_fact_check_results()
    print("Loaded fact-check results:", fact_check_results)
    
print("Verification results:")
if results and 'anomaly_results' in results:
    for model in results['anomaly_results']:
        print(model, results['anomaly_results'][model])