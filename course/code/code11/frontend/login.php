<?php
session_start(); // Start session

$correct_password = "xxxxxxx"; // Change this!

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    if (isset($_POST['password']) && $_POST['password'] === $correct_password) {
        $_SESSION['logged_in'] = true; // Store login status
        header("Location: index.php"); // Redirect to main page
        exit();
    } else {
        $error = "❌ Incorrect password. Try again.";
    }
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
        input, button { padding: 10px; margin: 5px; font-size: 16px; }
        .error { color: red; }
    </style>
</head>
<body>
    <h2>Login</h2>
    <?php if (isset($error)) echo "<p class='error'>$error</p>"; ?>
    <form method="POST">
        <input type="password" name="password" placeholder="Enter Password" required>
        <button type="submit">Login</button>
    </form>
</body>
</html>
