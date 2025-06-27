from PipelineManager import PipelineManager

config = {
    "data_source": "DisasterTweets.csv"
}
pipeline = PipelineManager(config)

verification = pipeline.run()

# Retrieve the fact-check results
if hasattr(pipeline, 'fact-checker'):
    results = pipeline.fact_checker.load_fact_check_results()
    print("Loaded fact-check results:", results)

print("Verification results:")
for model in verification:
    print(model, verification[model])