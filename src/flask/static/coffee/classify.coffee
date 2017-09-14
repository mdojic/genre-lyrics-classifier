class Classify 

	constructor: () ->

		@pageContainer = $(".classify--page--container")

		@classifyButton = @pageContainer.find(".classify--button")
		@lyricsTextArea = @pageContainer.find(".lyrics--text--area")

		@classifyButton.on "click", @_classifyClicked.bind( this )
		console.log ">>> @classifyButton = ", @classifyButton
		console.log ">>> @lyricsTextArea = ", @lyricsTextArea


	_classifyClicked: ( e ) ->
		console.log "> > > _classifyClicked"
		url    = "/get_genre"
		lyrics = @lyricsTextArea.val()
		dataForPosting = {
			lyrics: lyrics
		}
		
		AjaxUtils.sendAjax( 'POST', url, this, @_genreLoaded, @_genreLoadingError, dataForPosting, null )


	_genreLoaded: ( data, onTheFlyData ) ->
		console.log "Genre loaded: ", data

	_genreLoadingError: ( errorCode, errorMessage ) ->
		console.log "Error while classifying lyrics"
		console.log errorMessage
		console.log errorCode


$ ->
	new Classify()