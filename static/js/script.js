$(document).ready(function() {
    // Funcion de cargar archivo
    $("#uploadForm").on("submit", function(event) {
        event.preventDefault();

        var formData = new FormData();
        var imageInput = $("#imageInput")[0].files[0];
        formData.append("file", imageInput);

        $.ajax({
            url: "/predecir-imagen",
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $("#result").html(`<p>Emoción Predicha: ${response.prediction}</p>`);
            }
        });
    });
});