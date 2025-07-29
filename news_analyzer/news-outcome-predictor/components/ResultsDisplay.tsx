import React from 'react';
import type { AnalysisResult, GroundingSource } from '../types';
import { SourceList } from './SourceList';

interface ResultsDisplayProps {
  result: AnalysisResult;
  sources: GroundingSource[];
}

const PersonalizedIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M17.982 18.725A7.488 7.488 0 0 0 12 15.75a7.488 7.488 0 0 0-5.982 2.975m11.963 0a9 9 0 1 0-11.963 0m11.963 0A8.966 8.966 0 0 1 12 21a8.966 8.966 0 0 1-5.982-2.275M15 9.75a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
    </svg>
);

const SummaryIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
  </svg>
);

const KeyPointsIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 6.75h7.5M8.25 12h7.5m-7.5 5.25h7.5M3.75 6.75h.007v.008H3.75V6.75Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0ZM3.75 12h.007v.008H3.75V12Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm-.375 5.25h.007v.008H3.75v-.008Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z" />
  </svg>
);

const SentimentIcon: React.FC<{ sentiment: string; className: string }> = ({ sentiment, className }) => {
  if (sentiment === 'Positive') return <svg className={className} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M15.182 15.182a4.5 4.5 0 0 1-6.364 0M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0ZM9 9.75h.008v.008H9V9.75Zm6 0h.008v.008H15V9.75Z" /></svg>;
  if (sentiment === 'Negative') return <svg className={className} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M15.182 9.971A4.5 4.5 0 0 1 12 12.75a4.5 4.5 0 0 1-5.182-2.779M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0ZM9 9.75h.008v.008H9V9.75Zm6 0h.008v.008H15V9.75Z" /></svg>;
  return <svg className={className} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 5.25h.008v.008H12v-.008Z" /></svg>;
};

const PredictionIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 17.25v1.007a3 3 0 0 1-.97-2.122L7.5 15.5h9l-.53-1.622a3 3 0 0 1-.97-2.122v-1.007M12 15.75a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm0 0H9m3 0h3m-3 0V7.5M6 7.5h12M6 7.5a2.25 2.25 0 0 1-2.25-2.25V3.75A2.25 2.25 0 0 1 6 1.5h12a2.25 2.25 0 0 1 2.25 2.25v1.5A2.25 2.25 0 0 1 18 7.5M6 7.5v1.5M18 7.5v1.5" />
  </svg>
);


export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ result, sources }) => {
  const sentimentColor =
    result.sentiment === 'Positive'
      ? 'text-green-400'
      : result.sentiment === 'Negative'
      ? 'text-red-400'
      : 'text-yellow-400';

  return (
    <div className="space-y-6 animate-slide-up">
      {result.personalizedImpact && (
        <div className="bg-gradient-to-br from-purple-900/50 to-indigo-900/50 p-6 rounded-lg border border-purple-700 shadow-lg">
          <h3 className="flex items-center text-xl font-semibold text-purple-300 mb-3">
            <PersonalizedIcon className="h-6 w-6 mr-3" />
            Personalized Impact
          </h3>
          <p className="text-gray-200 text-lg leading-relaxed">{result.personalizedImpact}</p>
        </div>
      )}

      <div className="bg-gray-800/50 backdrop-blur-sm p-6 rounded-lg border border-gray-700">
        <h3 className="flex items-center text-xl font-semibold text-blue-300 mb-3">
          <SummaryIcon className="h-6 w-6 mr-3" />
          Summary
        </h3>
        <p className="text-gray-300 leading-relaxed">{result.summary}</p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-gray-800/50 backdrop-blur-sm p-6 rounded-lg border border-gray-700">
          <h3 className="flex items-center text-xl font-semibold text-blue-300 mb-3">
            <KeyPointsIcon className="h-6 w-6 mr-3" />
            Key Points
          </h3>
          <ul className="space-y-2 list-disc list-inside text-gray-300">
            {result.keyPoints.map((point, index) => (
              <li key={index}>{point}</li>
            ))}
          </ul>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm p-6 rounded-lg border border-gray-700">
          <h3 className="flex items-center text-xl font-semibold text-blue-300 mb-3">
            <SentimentIcon sentiment={result.sentiment} className="h-6 w-6 mr-3" />
            Sentiment
          </h3>
          <p className={`text-2xl font-bold ${sentimentColor}`}>{result.sentiment}</p>
        </div>
      </div>

      <div className="bg-gradient-to-br from-blue-900/50 to-indigo-900/50 p-6 rounded-lg border border-blue-700 shadow-lg">
         <h3 className="flex items-center text-xl font-semibold text-blue-300 mb-3">
          <PredictionIcon className="h-6 w-6 mr-3" />
          Outcome Prediction
        </h3>
        <p className="text-gray-200 text-lg leading-relaxed">{result.prediction}</p>
      </div>
      
      {sources.length > 0 && <SourceList sources={sources} />}
    </div>
  );
};