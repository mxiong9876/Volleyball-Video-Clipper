Volleyball Match Breakdown: App Blueprint

Core purpose: A web app that takes a YouTube link to a volleyball match, cuts out the dead time, and turns the match into a navigable timeline of rallies. Users can then jump to any point, follow the momentum, and export highlight reels.

1. Executive Summary

Value proposition

A 2-hour match usually contains only 20–30 minutes of live play. This app finds that play automatically and makes it searchable.
Paste a link and get back a scrubbable timeline of every rally, a point-by-point momentum chart, and one-click highlight reels.

Target audience

Players who want to rewatch their own points without scrubbing through timeouts.
Club and high school coaches who review film but can't afford Hudl-tier tools or hours of manual tagging.
Fans and content creators who want quick highlight cuts from long broadcasts.
Scouts and recruits: club players build recruiting reels from full-match uploads.

Differentiator

Tools like Hudl and Balltime depend on uploads from their own ecosystem, often with manual tagging. This app works from any public link, with no tagging needed.
2. Key Feature List

MVP (v1)

Link ingestion: paste a YouTube URL; the backend downloads the video, extracts audio, and queues analysis.
Rally segmentation: detect rally start and end times using whistle detection, motion energy, and scene-cut filtering, which removes replays and crowd shots.
Rally timeline viewer: a list or timeline of rallies. Clicking one plays that exact segment, and there's a "skip to next rally" mode.
Scoreboard OCR: the user drags a box around the score graphic once. The app then reads the score every few seconds and attaches it to each rally.
Momentum graph: point differential over time for each set, with runs highlighted. Clicking any point jumps to its clip.
Basic filters: by set, rally length, and score situation (e.g., "within 2 points").
Manual correction: users can nudge rally boundaries or fix a misread score. These corrections also become training labels later.

Future updates (v2+)

Highlight export: rank rallies by length and intensity (audio loudness, motion), then render an MP4 reel.
Learned rally detector: replace the heuristics with a classifier trained on user corrections.
Auto-detect the scoreboard region instead of asking the user to draw the box.
Bookmarks, tags, and notes on rallies, plus shareable links to a specific rally.
Team accounts so a coach can share a film library with players.
Phone-footage support: handle upload files with no scoreboard by inferring points from rally endings.
Bridge to the CV track: court homography, ball tracking, and a stat sheet built on the rally segments.
3. User Flow / Journey
Landing: the user opens the app and pastes a YouTube link, or picks from recently analyzed matches.
Preview: the app shows the title, thumbnail, and length. The user confirms, optionally entering team names.
Scoreboard setup (optional): the app shows one frame and the user drags a box over the score graphic. They can skip this step if the video has no scoreboard.
Processing: a progress screen shows the pipeline stages (downloading → extracting audio → finding rallies → reading scores). The user can leave and get notified when it's done.
Match overview: the user sees a summary: the number of rallies, live-play time vs. total time, and a momentum chart for each set.
Browse: the user scrolls the rally list, filters (e.g., "Set 3, rallies over 10 seconds"), or clicks a point on the momentum chart.
Watch: the player jumps straight to that rally. "Continuous mode" auto-advances through rallies, skipping all dead time.
Correct: if a boundary is off, the user drags the edges or fixes the score inline.
Save or share: the user bookmarks rallies or copies a link to one (v2: exports a highlight reel).
4. Recommended Tech Stack

Frontend

React + Vite + TypeScript: fast to build, and a good fit for a single-page app.
Tailwind CSS for styling.
YouTube IFrame Player API for playback. It plays the original video with start/end timestamps, so you never re-host footage in the MVP. This keeps you on safer ground with copyright and saves storage costs.
Recharts for the momentum graph.
TanStack Query for polling job status and fetching data.

Backend

FastAPI (Python): keeps the API in the same language as the analysis code.
Task queue: Celery or RQ with Redis, because analysis takes minutes and can't run inside a web request.
Analysis libraries:
yt-dlp to download the video.
ffmpeg to extract audio and cut clips.
PySceneDetect to detect camera cuts and replays.
librosa / scipy to detect whistles (look for a narrow-band tone around 2–4 kHz).
OpenCV for frame differencing and motion energy.
PaddleOCR or EasyOCR to read the scoreboard.

Database

PostgreSQL, managed through Supabase, which also provides auth and file storage. The data is relational (videos → sets → rallies → scores), so SQL is a natural fit.

Hosting

Frontend: Vercel or Netlify.
API: Railway, Render, or Fly.io.
Workers: a CPU worker covers the MVP, since the heuristics and OCR are light. Move to Modal or a GPU box once you add learned models.
Object storage (v2, for exported reels): Cloudflare R2 or Supabase Storage.

Legal note: downloading YouTube videos conflicts with YouTube's Terms of Service. That's fine for a personal project or portfolio demo. For a public product, a safer design is to analyze only videos users upload themselves, or content they own, while still using the embed player for playback.

5. Database Schema Outline
users: id, email, display_name, created_at
videos: id, youtube_id, title, duration_sec, submitted_by → users.id, scoreboard_roi (JSON: x, y, w, h), team_a_name, team_b_name, created_at
Make youtube_id unique so repeat submissions reuse the existing analysis.
analysis_jobs: id, video_id → videos.id, status (queued / downloading / segmenting / ocr / done / failed), progress_pct, pipeline_version, error_msg, started_at, finished_at
Store pipeline_version so you can re-run old videos when the detector improves.
sets: id, video_id → videos.id, set_number, start_sec, end_sec, final_score_a, final_score_b
rallies: id, video_id → videos.id, set_id → sets.id, rally_index, start_sec, end_sec, duration_sec, confidence, intensity_score, is_user_corrected
score_snapshots: id, rally_id → rallies.id, score_a, score_b, point_winner (a / b / unknown), ocr_confidence
Read the score after each rally ends; the change from the previous snapshot tells you who won the point.
bookmarks: id, user_id → users.id, rally_id → rallies.id, note, tags (array), created_at
corrections: id, user_id, rally_id, field (start / end / score), old_value, new_value, created_at
This table is your future training dataset. Keep it from day one.

Relationships

One user submits many videos.
A video has many jobs, sets, and rallies.
A set has many rallies.
A rally has one score snapshot and many bookmarks and corrections.
6. Development Phases

Phase 0: Setup (≈ 3 days)

Create the monorepo (/web, /api, /pipeline), set up the Supabase project, and get Redis running locally with Docker Compose.
Collect 5–10 test matches. Include at least one broadcast and one amateur video.
Hand-label rally boundaries for 2–3 of those matches. This is your evaluation set; without it you can't tell whether a change actually helped.

Phase 1: Ingestion pipeline (≈ 1 week)

Build a worker task that turns a URL into a downloaded video plus extracted audio and metadata in the database.
Add job status updates and a basic API: POST /videos, GET /jobs/:id.

Phase 2: Rally segmentation (≈ 2 weeks — the core of the product)

Start with an audio-only whistle detector and measure it against your labels.
Add motion energy from frame differencing, then scene-cut detection to drop replays.
Combine the signals with simple rules: a rally starts at a serve whistle and ends when motion drops or the next whistle sounds.
Evaluate with precision and recall on rally boundaries, using a ±1–2 second tolerance. Aim for roughly 85% or better before moving on.

Phase 3: Scoreboard OCR + momentum (≈ 1 week)

Run OCR inside the user's box, sampling at each rally's end.
Add sanity logic: scores only increase by 0 or 1 per rally, and they reset between sets. This fixes most OCR misreads.
Derive set boundaries and point winners from the score changes.

Phase 4: Frontend viewer (≈ 2 weeks)

Build the paste-link page, the scoreboard box selector, the processing screen, and the match overview.
Add the rally list with the embedded player, continuous mode, the momentum chart with click-to-jump, and filters.
Add inline correction tools that write to the corrections table.

Phase 5: Polish + deploy (≈ 1 week)

Add auth, caching of repeat videos, and error states (private video, too long, no scoreboard).
Deploy the frontend, API, and worker, and put a cap on video length to control compute costs.
Write a README with your evaluation numbers. Measured accuracy makes the project far stronger on a resume than "it works."

Phase 6: v2 (ongoing)

Highlight ranking and MP4 export, auto-detection of the scoreboard, and a learned rally classifier trained on your labels plus user corrections.
Then move into the computer vision track: court homography, ball tracking, and touch-level stats.
