
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import type { AnalysisResult, GroundingSource, UserProfile } from '../types';

const findJson = (text: string): object | null => {
    const jsonRegex = /```json\s*([\s\S]*?)\s*```/;
    const match = text.match(jsonRegex);
    if (match && match[1]) {
        try {
            return JSON.parse(match[1]);
        } catch (e) {
            console.error("Failed to parse JSON from markdown block:", e);
            return null;
        }
    }
    // Fallback for raw JSON
    try {
        return JSON.parse(text);
    } catch (e) {
        console.error("Failed to parse raw text as JSON:", e);
    }
    return null;
}

export const analyzeAndPredictNews = async (
  newsTopic: string,
  userProfile: UserProfile
): Promise<{ analysis: AnalysisResult; sources: GroundingSource[] } | null> => {
  if (!process.env.API_KEY) {
    throw new Error("API_KEY environment variable is not configured.");
  }
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

  const prompt = `
    Analyze the following news topic or article based on the user's profile.

    USER PROFILE:
    - Profession: ${userProfile.profession}
    - Location: ${userProfile.location}
    - Interests: ${userProfile.interests}

    NEWS TOPIC:
    "${newsTopic}"
    
    Perform the following actions:
    1.  Provide a concise summary of the main points.
    2.  Extract 3-5 key bullet points.
    3.  Determine the overall sentiment (Positive, Negative, or Neutral).
    4.  Based on the information, provide a likely prediction of the outcome or future implications.
    5.  **Personalized Impact**: Based on the user's profile, provide a specific, personalized analysis of how this news could directly impact them. Explain potential effects on their profession, location, or interests.

    You MUST respond with only a single, valid JSON object enclosed in a markdown code block (\`\`\`json ... \`\`\`). Do not include any text before or after the JSON block.
    The JSON object must have the following structure:
    {
      "summary": "string",
      "keyPoints": ["string"],
      "sentiment": "string (Positive, Negative, or Neutral)",
      "prediction": "string",
      "personalizedImpact": "string"
    }
  `;

  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        tools: [{ googleSearch: {} }],
        temperature: 0.3,
      },
    });

    const rawText = response.text;
    const analysis = findJson(rawText) as AnalysisResult | null;
    
    if (!analysis) {
        console.error("Could not parse JSON from response:", rawText);
        throw new Error("The AI response was not in the expected format. Please try again.");
    }
    
    const rawSources = response.candidates?.[0]?.groundingMetadata?.groundingChunks ?? [];
    const sources: GroundingSource[] = rawSources
      .map((chunk: any) => chunk?.web)
      .filter((web: any) => web && web.uri)
      .map((web: any) => ({ uri: web.uri, title: web.title || 'Untitled Source' }));

    return { analysis, sources };

  } catch (error) {
    console.error("Error calling Gemini API:", error);
    throw new Error("Failed to communicate with the AI service. Check your connection and API key.");
  }
};