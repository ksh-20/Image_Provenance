import { type NextRequest, NextResponse } from "next/server";

// Free backend configuration
const FREE_API_URL =
  process.env.FREE_DEEPFAKE_API_URL || "http://localhost:5000";

interface FreeAPIResponse {
  prediction: string; // "likely authentic", "possibly manipulated", "likely deepfake"
  confidence: number;
  probability: number;
  explanation: string;
  issues_found?: string[];
  details: {
    sharpness_score: number;
    compression_artifacts: number;
    color_consistency: number;
    lighting_consistency: number;
    faces_detected: number;
  };
}

interface ReverseImageResult {
  url: string;
  title: string;
  source: string;
}

interface AnalysisResult {
  verdict: string;
  confidence: number;
  probability: number;
  explanation: string;
  issuesFound?: string[];
  riskLevel: "low" | "medium" | "high";
  details: {
    sharpness: number;
    compressionArtifacts: number;
    colorConsistency: number;
    lightingConsistency: number;
    facesDetected: number;
  };
  similarImages?: ReverseImageResult[];
  reverseSearchUrl?: string;
  processingTime: string;
  filename: string;
  filesize: string;
}

// Free reverse image search using Google Custom Search API (free tier: 100 queries/day)
async function searchSimilarImages(
  imageUrl: string
): Promise<ReverseImageResult[]> {
  try {
    // Check if API keys are set (optional, falls back to manual search)
    const apiKey = process.env.GOOGLE_CSE_API_KEY;
    const searchEngineId = process.env.GOOGLE_CSE_ID;

    if (!apiKey || !searchEngineId) {
      // Return fallback option to open reverse image search manually
      return [
        {
          url: `https://www.google.com/searchbyimage?image_url=${encodeURIComponent(
            imageUrl
          )}`,
          title: "Search similar images on Google",
          source: "Google Reverse Image Search",
        },
        {
          url: `https://tineye.com/search?url=${encodeURIComponent(imageUrl)}`,
          title: "Search similar images on TinEye",
          source: "TinEye Reverse Image Search",
        },
        {
          url: `https://yandex.com/images/search?url=${encodeURIComponent(
            imageUrl
          )}&rpt=imageview`,
          title: "Search similar images on Yandex",
          source: "Yandex Reverse Image Search",
        },
      ];
    }

    // Use Google Custom Search API (free tier)
    const searchUrl = `https://www.googleapis.com/customsearch/v1?key=${apiKey}&cx=${searchEngineId}&searchType=image&q=${encodeURIComponent(
      imageUrl
    )}&num=3`;

    const response = await fetch(searchUrl);
    if (!response.ok) {
      throw new Error("Search API failed");
    }

    const data = await response.json();

    if (data.items && data.items.length > 0) {
      return data.items.slice(0, 3).map((item: any) => ({
        url: item.link,
        title: item.title,
        source: item.displayLink,
      }));
    }

    return [];
  } catch (error) {
    console.error("Reverse image search failed:", error);
    // Return fallback manual search links
    return [
      {
        url: `https://www.google.com/searchbyimage?image_url=${encodeURIComponent(
          imageUrl
        )}`,
        title: "Search similar images on Google",
        source: "Google Reverse Image Search",
      },
    ];
  }
}

export async function POST(request: NextRequest) {
  const startTime = Date.now();

  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json(
        {
          success: false,
          error: "No file provided",
        },
        { status: 400 }
      );
    }

    // Validate file type (images only)
    const supportedTypes = [
      "image/jpeg",
      "image/jpg",
      "image/png",
      "image/webp",
    ];

    if (!supportedTypes.includes(file.type)) {
      return NextResponse.json(
        {
          success: false,
          error: "Only image files (JPG, PNG, WebP) are supported",
        },
        { status: 400 }
      );
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      return NextResponse.json(
        {
          success: false,
          error: "File size too large (max 10MB)",
        },
        { status: 400 }
      );
    }

    const filename = file.name;
    const filesize = `${(file.size / 1024 / 1024).toFixed(2)} MB`;

    // Forward to free backend
    const backendFormData = new FormData();
    backendFormData.append("file", file);

    const backendResponse = await fetch(`${FREE_API_URL}/analyze`, {
      method: "POST",
      body: backendFormData,
    });

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      console.error("Backend error:", backendResponse.status, errorText);

      return NextResponse.json(
        {
          success: false,
          error: `Analysis failed: ${backendResponse.statusText}`,
          details: errorText,
        },
        { status: backendResponse.status }
      );
    }

    const backendResult: FreeAPIResponse = await backendResponse.json();
    const processingTime = `${((Date.now() - startTime) / 1000).toFixed(1)}s`;

    // Determine risk level
    let riskLevel: "low" | "medium" | "high" = "low";
    if (backendResult.prediction === "likely deepfake") {
      riskLevel = "high";
    } else if (backendResult.prediction === "possibly manipulated") {
      riskLevel = "medium";
    }

    // Provide reverse image search options
    const similarImages: ReverseImageResult[] = [
      {
        url: "https://lens.google.com/uploadbyurl",
        title: "Search with Google Lens (drag & drop your image)",
        source: "Google Lens"
      },
      {
        url: "https://tineye.com/",
        title: "Search on TinEye (drag & drop your image)",
        source: "TinEye"
      },
      {
        url: "https://yandex.com/images/",
        title: "Search on Yandex (click camera icon, upload image)",
        source: "Yandex Images"
      }
    ];
    
    // Create instruction for reverse search
    const reverseSearchUrl = "https://lens.google.com/";

    // Format response
    const analysis: AnalysisResult = {
      verdict: backendResult.prediction,
      confidence: Math.round(backendResult.confidence * 100),
      probability: Math.round(backendResult.probability * 100),
      explanation: backendResult.explanation,
      issuesFound: backendResult.issues_found || [],
      riskLevel,
      details: {
        sharpness: backendResult.details.sharpness_score,
        compressionArtifacts: backendResult.details.compression_artifacts,
        colorConsistency: backendResult.details.color_consistency,
        lightingConsistency: backendResult.details.lighting_consistency,
        facesDetected: backendResult.details.faces_detected,
      },
      similarImages,
      reverseSearchUrl,
      processingTime,
      filename,
      filesize,
    };

    return NextResponse.json({
      success: true,
      analysis,
    });
  } catch (error) {
    console.error("Analysis error:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Analysis failed",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
