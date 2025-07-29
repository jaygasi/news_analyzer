import React, { useState, useCallback } from 'react';
import { Header } from './components/Header';
import { UserProfile } from './components/UserProfile';
import { InputArea } from './components/InputArea';
import { ResultsDisplay } from './components/ResultsDisplay';
import { LoadingSpinner } from './components/LoadingSpinner';
import { ErrorMessage } from './components/ErrorMessage';
import { WelcomeMessage } from './components/WelcomeMessage';
import { analyzeAndPredictNews } from './services/geminiService';
import type { AnalysisResult, GroundingSource, UserProfile as UserProfileType } from './types';

const App: React.FC = () => {
  const [userProfile, setUserProfile] = useState<UserProfileType>({
    profession: 'Software Developer',
    location: 'San Francisco, USA',
    interests: 'Technology, AI, renewable energy, financial markets',
  });
  const [newsInput, setNewsInput] = useState<string>('');
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [sources, setSources] = useState<GroundingSource[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleProfileChange = useCallback((e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setUserProfile(prev => ({ ...prev, [name]: value }));
  }, []);

  const handleAnalysis = useCallback(async () => {
    if (!newsInput.trim()) {
      setError('Please enter a news topic or article to analyze.');
      return;
    }
    setIsLoading(true);
    setError(null);
    setAnalysis(null);
    setSources([]);

    try {
      const result = await analyzeAndPredictNews(newsInput, userProfile);
      if (result) {
        setAnalysis(result.analysis);
        setSources(result.sources);
      } else {
        setError('Failed to get a valid analysis from the AI. Please try again.');
      }
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred.');
    } finally {
      setIsLoading(false);
    }
  }, [newsInput, userProfile]);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 font-sans flex flex-col items-center p-4 sm:p-6 lg:p-8">
      <div className="w-full max-w-4xl">
        <Header />
        <main className="mt-8">
          <UserProfile
            profile={userProfile}
            onProfileChange={handleProfileChange}
            isDisabled={isLoading}
          />
          <InputArea
            value={newsInput}
            onChange={(e) => setNewsInput(e.target.value)}
            onSubmit={handleAnalysis}
            isLoading={isLoading}
          />
          {error && <ErrorMessage message={error} />}
          <div className="mt-8">
            {isLoading && <LoadingSpinner />}
            {!isLoading && !error && analysis && (
              <ResultsDisplay result={analysis} sources={sources} />
            )}
            {!isLoading && !error && !analysis && <WelcomeMessage />}
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
