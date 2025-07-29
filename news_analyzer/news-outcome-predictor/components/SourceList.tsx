
import React from 'react';
import type { GroundingSource } from '../types';

interface SourceListProps {
  sources: GroundingSource[];
}

const LinkIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M13.19 8.688a4.5 4.5 0 0 1 1.242 7.244l-4.5 4.5a4.5 4.5 0 0 1-6.364-6.364l1.757-1.757m13.35-.622 1.757-1.757a4.5 4.5 0 0 0-6.364-6.364l-4.5 4.5a4.5 4.5 0 0 0 1.242 7.244" />
  </svg>
);

export const SourceList: React.FC<SourceListProps> = ({ sources }) => {
  if (!sources || sources.length === 0) {
    return null;
  }
  
  return (
    <div className="bg-gray-800/50 backdrop-blur-sm p-6 rounded-lg border border-gray-700 animate-slide-up">
      <h3 className="flex items-center text-xl font-semibold text-blue-300 mb-4">
        <LinkIcon className="h-6 w-6 mr-3" />
        Information Sources
      </h3>
      <p className="text-gray-400 mb-4 text-sm">
        This analysis was informed by the following sources found via Google Search.
      </p>
      <ul className="space-y-3">
        {sources.map((source, index) => (
          <li key={index}>
            <a
              href={source.uri}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-start gap-3 p-3 -m-3 rounded-md hover:bg-gray-700/50 transition-colors"
            >
              <div className="flex-shrink-0 mt-1">
                <div className="w-2 h-2 rounded-full bg-brand-secondary"></div>
              </div>
              <div className="flex-grow">
                <p className="font-semibold text-gray-200 group-hover:text-brand-secondary transition-colors truncate">
                  {source.title || 'Untitled Source'}
                </p>
                <p className="text-xs text-gray-500 group-hover:text-gray-400 transition-colors truncate">
                  {source.uri}
                </p>
              </div>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
};
