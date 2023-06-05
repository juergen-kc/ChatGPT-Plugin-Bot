$(function() {
    $('#form').on('submit', function(e) {
        console.log("Form submission detected");
        try {
            e.preventDefault();
            console.log("Default form submission prevented");
            let question = $('#input').val();
            if (!question.trim()) {
                return;  // If question is empty or all whitespace, do not send it
            }
            $('#input').val('');
            $('#chat').append($('<p>').text('You: ' + question));
            $.ajax({
                url: '/ask',
                type: 'post',
                contentType: 'application/json',
                data: JSON.stringify({ query: question }),
                success: function(response) {
                    const data = JSON.parse(response);
                    $('#chat').append($('<p>').text('AI: ' + data.answer));
                    $('#chat').scrollTop($('#chat')[0].scrollHeight);
                },
                error: function(response) {
                    $('#chat').append($('<p>').text('AI: Sorry, I am unable to process your request at the moment.'));
                    $('#chat').scrollTop($('#chat')[0].scrollHeight);
                }
            });
        } catch (error) {
            console.error("An error occurred during form submission:", error);
        }
    });
});
