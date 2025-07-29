
import React from 'react';

interface ErrorMessageProps {
  message: string;
}

const ErrorIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
  </svg>
);


export const ErrorMessage: React.FC<ErrorMessageProps> = ({ message }) => {
  return (
    <div className="mt-6 flex items-start p-4 bg-red-900/30 border border-red-500/50 rounded-lg animate-fade-in" role="alert">
      <ErrorIcon className="h-6 w-6 text-red-400 mr-3 flex-shrink-0 mt-0.5" />
      <div>
        <h4 className="font-semibold text-red-300">An Error Occurred</h4>
        <p className="text-red-400">{message}</p>
      </div>
    </div>
  );
};
