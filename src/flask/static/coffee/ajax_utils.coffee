class @AjaxUtils

	@sendAjax: ( method, url, callbackContext, callbackSuccess, callbackError, data, onTheFlyData ) ->

		# console.log ":: :: :: sendAjax"
		# console.log ":: method = ", method 
		# console.log ":: url = ", url 
		# console.log ":: callbackContext = ", callbackContext
		# console.log ":: callbackSuccess = ", callbackSuccess
		# console.log ":: callbackError = ", callbackError 
		# console.log ":: data = ", data 
		# console.log ":: onTheFlyData = ", onTheFlyData

		if (not data?) or (data is undefined)
			data = ""

		$.ajax
			url         : url
			type        : method
			crossDomain : true
			data        : data

			success : ( response ) ->
				callbackSuccess.call( callbackContext, response, onTheFlyData )

			error : (xhr, status) ->
				callbackError.call( callbackContext, xhr, status )
