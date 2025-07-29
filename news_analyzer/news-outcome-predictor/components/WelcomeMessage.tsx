
import React from 'react';

const InfoIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z" />
    </svg>
);


export const WelcomeMessage: React.FC = () => {
  return (
    <div className="text-center p-8 bg-gray-800/50 rounded-lg border border-gray-700 animate-fade-in">
        <InfoIcon className="h-12 w-12 mx-auto text-brand-secondary mb-4"/>
      <h2 className="text-2xl font-bold text-gray-100">Welcome to the News Analyzer</h2>
      <p className="mt-2 text-gray-400 max-w-2xl mx-auto">
        Paste a news article, topic, or URL into the text box above. The AI will provide a detailed analysis, including a summary, key points, sentiment, and a predicted outcome based on current information from the web.
      </p>
       <div className="mt-6 text-sm text-gray-500">
        <p>Example: "What is the future of remote work post-2025?"</p>
      </div>
    </div>
  );
};
