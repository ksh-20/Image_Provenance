-- Insert sample users
INSERT INTO users (username, email, password_hash, profile_pic, bio, is_verified) VALUES
('alice_doe', 'alice@example.com', '$2b$12$hash1', '/placeholder.svg?height=40&width=40', '📸 Photography enthusiast | 🌍 Travel lover', true),
('john_smith', 'john@example.com', '$2b$12$hash2', '/placeholder.svg?height=40&width=40', '🍳 Chef | 📱 Food blogger', false),
('sarah_wilson', 'sarah@example.com', '$2b$12$hash3', '/placeholder.svg?height=40&width=40', '🎨 Digital artist | ✨ Creative soul', true),
('mike_johnson', 'mike@example.com', '$2b$12$hash4', '/placeholder.svg?height=40&width=40', '🏃‍♂️ Fitness coach | 💪 Motivation daily', false);

-- Insert sample posts
INSERT INTO posts (user_id, media_url, media_type, caption, deepfake_score, likes_count, comments_count) VALUES
(1, '/placeholder.svg?height=600&width=600', 'image', 'Beautiful sunset today! 🌅 #nature #photography', 15, 234, 12),
(2, '/placeholder.svg?height=600&width=600', 'image', 'Homemade pasta night! 🍝 Recipe in my bio', 8, 156, 23),
(3, '/placeholder.svg?height=600&width=600', 'image', 'New digital artwork completed! What do you think? 🎨', 72, 89, 15),
(4, '/placeholder.svg?height=600&width=600', 'image', 'Morning workout session! 💪 #fitness #motivation', 12, 201, 8);

-- Insert sample stories
INSERT INTO stories (user_id, media_url, media_type, deepfake_score, expires_at, view_count) VALUES
(1, '/placeholder.svg?height=600&width=400', 'image', 25, NOW() + INTERVAL '24 hours', 45),
(2, '/placeholder.svg?height=600&width=400', 'image', 18, NOW() + INTERVAL '20 hours', 32),
(3, '/placeholder.svg?height=600&width=400', 'image', 91, NOW() + INTERVAL '18 hours', 67);

-- Insert sample follows
INSERT INTO follows (follower_id, following_id) VALUES
(1, 2), (1, 3), (1, 4),
(2, 1), (2, 3),
(3, 1), (3, 2), (3, 4),
(4, 1), (4, 2);

-- Insert sample likes
INSERT INTO likes (user_id, post_id) VALUES
(2, 1), (3, 1), (4, 1),
(1, 2), (3, 2), (4, 2),
(1, 3), (2, 3),
(1, 4), (2, 4), (3, 4);

-- Insert sample comments
INSERT INTO comments (user_id, post_id, content) VALUES
(2, 1, 'Absolutely stunning! 😍'),
(3, 1, 'Great composition!'),
(4, 1, 'Love the colors in this shot'),
(1, 2, 'Looks delicious! Can you share the recipe?'),
(3, 2, 'Making me hungry! 🤤'),
(1, 3, 'Your art style is incredible'),
(3, 2, 'Making me hungry! 🤤'),
(1, 3, 'Your art style is incredible! Keep it up! 🎨'),
(2, 3, 'This would look great as a print'),
(1, 4, 'Great form! What''s your workout routine?'),
(2, 4, 'Inspiring as always! 💪');

-- Insert sample messages
INSERT INTO messages (sender_id, receiver_id, content, is_read) VALUES
(1, 2, 'Hey! How are you doing?', true),
(2, 1, 'I''m doing great! Just posted a new recipe.', true),
(1, 2, 'I saw it! Looks amazing 🤤', false),
(3, 1, 'Love your latest photo!', true),
(1, 3, 'Thank you so much! 😊', true),
(4, 1, 'Want to collaborate on a fitness shoot?', false);

-- Insert sample deepfake analyses
INSERT INTO deepfake_analyses (post_id, confidence_score, model_version, analysis_details, processing_time_ms) VALUES
(1, 15, 'v2.1.0', '{"face_consistency": 92, "temporal_consistency": 88, "artifact_detection": 95}', 2300),
(2, 8, 'v2.1.0', '{"face_consistency": 96, "temporal_consistency": 94, "artifact_detection": 98}', 1800),
(3, 72, 'v2.1.0', '{"face_consistency": 45, "temporal_consistency": 38, "artifact_detection": 52}', 2100),
(4, 12, 'v2.1.0', '{"face_consistency": 89, "temporal_consistency": 91, "artifact_detection": 93}', 1950);

-- Insert sample story views
INSERT INTO story_views (story_id, viewer_id) VALUES
(1, 2), (1, 3), (1, 4),
(2, 1), (2, 3),
(3, 1), (3, 2), (3, 4);
