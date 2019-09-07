
// The query model
function Query(selfHref, rawText, created, status, summary) {
    var self = this;
    self.selfHref = selfHref;
    self.rawText = ko.observable(rawText);
    self.created = created;
    self.status = status;
    self.summary = summary;
    self.editing = ko.observable(false);

    // Behaviors
    self.edit = function() { this.editing(true) }
}

// The query view model
function QueryViewModel() {
    var self = this;

    self.newRawText = ko.observable();
    self.querys = ko.observableArray([]);


    // add query: send POST to querys resource
    self.addQuery = function () {
        // a little bit of pre-processing of user entered rawText 
        var newRawText = self.newRawText();
        console.log(newRawText)
        if (typeof newRawText == "undefined") {
            alert("RawText required");
            return;
        }
        if (newRawText == ""){
            alert("RawText required");
            return
        }

        // TODO some raw text input validations



        // make POST request
        $.ajax("http://localhost:8080/predict", {
            data: '{"words": "' + newRawText +  '"}',
            type: "post",
            contentType: "application/json",
            success: function (data) {
                var query = new Query(self.querys.length, newRawText, new Date(), data.confidence, data.result)
                self.querys.push(query)
                //self.loadQuerys();
                self.newRawText("");
            }
        });
    };

    // update query: send PUT to existing querys resource
    self.updateQuery = function (query) {

        // same as in "addQuery" a little bit of parameter checking. Some code duplication here
        // but we leave it for demonstration purposes
        var thisquery =query;
        var newRawText = query.rawText();
        if (typeof newRawText == "undefined") {
            alert("RawText required");
            return;
        }
        if (newRawText == ""){
            self.deleteQuery(thisquery);
            return
        }

        // TODO add input validation here


        // make PUT request (or send PATCH then we don't need to include the created date)
        $.ajax("http://localhost:8080/predict", {
            data: '{"words": "' + newRawText +'"}',
            type: "post",
            contentType: "application/json",
            success: function (data) {
                self.deleteQuery(thisquery);
                var query = new Query(self.querys.length, newRawText, new Date(), data.confidence, data.result)
                self.querys.push(query)
            }
        });
    };


    // delete query: send DELETE to querys resource
    self.deleteQuery = function (query) {
            self.querys.splice(query.selfHref, 1);
    };

}




ko.applyBindings(new QueryViewModel());