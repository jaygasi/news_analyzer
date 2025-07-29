import React from 'react';
import type { UserProfile as UserProfileType } from '../types';

interface UserProfileProps {
  profile: UserProfileType;
  onProfileChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => void;
  isDisabled: boolean;
}

const UserIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
  </svg>
);

export const UserProfile: React.FC<UserProfileProps> = ({ profile, onProfileChange, isDisabled }) => {
    return (
        <div className="mb-8 p-6 bg-gray-800/50 rounded-lg border border-gray-700 animate-fade-in">
            <h2 className="flex items-center text-xl font-semibold text-blue-300 mb-4">
                <UserIcon className="h-6 w-6 mr-3" />
                Your Profile for Personalization
            </h2>
            <p className="text-sm text-gray-400 mb-4">
                Provide some details about yourself to get a personalized impact analysis. This information is only used for your current session.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label htmlFor="profession" className="block text-sm font-medium text-gray-300 mb-1">Profession</label>
                    <input
                        type="text"
                        id="profession"
                        name="profession"
                        value={profile.profession}
                        onChange={onProfileChange}
                        disabled={isDisabled}
                        className="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-gray-200 focus:ring-1 focus:ring-brand-secondary focus:border-brand-secondary transition disabled:opacity-50"
                        placeholder="e.g., Doctor, Artist, Engineer"
                    />
                </div>
                 <div>
                    <label htmlFor="location" className="block text-sm font-medium text-gray-300 mb-1">Location</label>
                    <input
                        type="text"
                        id="location"
                        name="location"
                        value={profile.location}
                        onChange={onProfileChange}
                        disabled={isDisabled}
                        className="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-gray-200 focus:ring-1 focus:ring-brand-secondary focus:border-brand-secondary transition disabled:opacity-50"
                        placeholder="e.g., New York, USA"
                    />
                </div>
                <div className="md:col-span-2">
                     <label htmlFor="interests" className="block text-sm font-medium text-gray-300 mb-1">Interests</label>
                     <textarea
                        id="interests"
                        name="interests"
                        value={profile.interests}
                        onChange={onProfileChange}
                        disabled={isDisabled}
                        rows={3}
                        className="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-gray-200 focus:ring-1 focus:ring-brand-secondary focus:border-brand-secondary transition resize-y disabled:opacity-50"
                        placeholder="e.g., Climate change, startups, local politics"
                     />
                </div>
            </div>
        </div>
    );
}
