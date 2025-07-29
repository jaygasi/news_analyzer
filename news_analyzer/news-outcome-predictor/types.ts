export interface UserProfile {
  profession: string;
  location: string;
  interests: string;
}

export interface AnalysisResult {
  summary: string;
  keyPoints: string[];
  sentiment: 'Positive' | 'Negative' | 'Neutral' | string;
  prediction: string;
  personalizedImpact: string;
}

export interface GroundingSource {
  uri: string;
  title: string;
}
