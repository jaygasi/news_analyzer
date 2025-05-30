from utils.env_utils import read_env_variable
from universe_selection.universe_selector import UniverseSelector
from analysis_tools.phase_keyword_analyzer import PhaseKeywordAnalyzer
from analysis_tools.topic_keyword_analyzer import TopicKeywordAnalyzer

# Get API keys
FMP_API_KEY = read_env_variable('FMP_API_KEY')


# Initialize event trackers and processors
universe_selector = UniverseSelector(FMP_API_KEY)
universe_selector.perform_selection()


if __name__ == "__main__":
    # Perform keyword analysis:
    #phase_keyword_analyzer = PhaseKeywordAnalyzer(FMP_API_KEY)
    #phase_keyword_analyzer.perform_analysis()
    topic_keyword_analyzer = TopicKeywordAnalyzer(FMP_API_KEY)
    topic_keyword_analyzer.perform_analysis()

    print("All done. Review the keyword stats lists in the 'results' directory, then update the keyword lists in the news_topic_detector.py")
